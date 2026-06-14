"""Result classification ops: manifest, crop export, bundle, ResNet train/load, live corrections."""
from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from studio.config import (
    DATA_RESULT_CLS_DIR,
    RESULT_CLS_AUGMENTED_DIR,
    RESULT_CLS_AUGMENT_INDEX,
    RESULT_CLS_AUG_PER_STEM,
    RESULT_CLS_ARCH,
    RESULT_CLS_BUNDLE_DIR,
    RESULT_CLS_CROPS_DIR,
    RESULT_CLS_LIVE_LOG,
    RESULT_CLS_LIVE_SESSION_JSON,
    RESULT_CLS_MANIFEST,
    RESULT_CLS_RUNS_DIR,
    SCOPE_ALL_LIVE,
    SCOPE_CURRENT_LIVE,
    SCOPE_FULL,
    SCOPE_LABELED_ONLY,
    SUPPORTED_EXTENSIONS,
)
from studio.crop_ops import RESULT_CLS_IMGSZ, letterbox, order_quad_points, warp_quad_crop

RESULT_CLASSES = ("positive", "negative", "invalid")
SOURCE_LABELED = "labeled"
SOURCE_LIVE = "live"
_VAL_FRACTION = 0.15
_BUNDLE_SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ManifestEntry:
    stem: str
    cls: str
    source: str = SOURCE_LABELED
    session_id: str = "legacy"


# ─── Manifest I/O (v2: stem, class, source, session_id) ───────────────────────

def read_manifest_entries(path: Path | None = None) -> list[ManifestEntry]:
    path = path or RESULT_CLS_MANIFEST
    entries: list[ManifestEntry] = []
    if not path.is_file():
        return entries
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            stem = row[0].strip()
            if stem.lower() == "stem":
                continue
            cls = row[1].strip().lower()
            if cls not in RESULT_CLASSES or not stem:
                continue
            source = row[2].strip().lower() if len(row) > 2 and row[2].strip() else SOURCE_LABELED
            if source not in (SOURCE_LABELED, SOURCE_LIVE):
                source = SOURCE_LABELED
            session_id = row[3].strip() if len(row) > 3 and row[3].strip() else "legacy"
            entries.append(ManifestEntry(stem=stem, cls=cls, source=source, session_id=session_id))
    return entries


def read_manifest(path: Path | None = None) -> dict[str, str]:
    """stem → class (backward compatible)."""
    return {e.stem: e.cls for e in read_manifest_entries(path)}


def write_manifest_entries(entries: list[ManifestEntry], path: Path | None = None) -> None:
    path = path or RESULT_CLS_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries, key=lambda e: e.stem)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "class", "source", "session_id"])
        for e in sorted_entries:
            writer.writerow([e.stem, e.cls, e.source, e.session_id])


def write_manifest(manifest: dict[str, str], path: Path | None = None) -> None:
    """Write classes only; preserves source/session from existing rows when possible."""
    existing = {e.stem: e for e in read_manifest_entries(path)}
    out: list[ManifestEntry] = []
    for stem in sorted(manifest.keys()):
        cls = manifest[stem]
        if cls not in RESULT_CLASSES:
            continue
        prev = existing.get(stem)
        if prev:
            out.append(ManifestEntry(stem, cls, prev.source, prev.session_id))
        else:
            out.append(ManifestEntry(stem, cls, SOURCE_LABELED, "legacy"))
    write_manifest_entries(out, path)


def upsert_manifest_entry(entry: ManifestEntry, path: Path | None = None) -> None:
    entries = read_manifest_entries(path)
    by_stem = {e.stem: e for e in entries}
    by_stem[entry.stem] = entry
    write_manifest_entries(list(by_stem.values()), path)


def filter_entries_by_scope(
    entries: list[ManifestEntry],
    scope: str,
    live_session_id: str | None = None,
) -> list[ManifestEntry]:
    if scope == SCOPE_FULL:
        return list(entries)
    if scope == SCOPE_ALL_LIVE:
        return [e for e in entries if e.source == SOURCE_LIVE]
    if scope == SCOPE_LABELED_ONLY:
        return [e for e in entries if e.source == SOURCE_LABELED]
    if scope == SCOPE_CURRENT_LIVE:
        sid = live_session_id or ""
        return [e for e in entries if e.source == SOURCE_LIVE and e.session_id == sid]
    return list(entries)


def make_live_session_id() -> str:
    return time.strftime("live_%Y%m%d_%H%M%S")


def persist_live_session_id(session_id: str) -> None:
    """Remember last live session so Train tab works after app restart."""
    RESULT_CLS_LIVE_SESSION_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_CLS_LIVE_SESSION_JSON.write_text(
        json.dumps({"session_id": session_id, "updated": time.time()}),
        encoding="utf-8",
    )


def load_persisted_live_session_id() -> str | None:
    if not RESULT_CLS_LIVE_SESSION_JSON.is_file():
        return None
    try:
        data = json.loads(RESULT_CLS_LIVE_SESSION_JSON.read_text(encoding="utf-8"))
        sid = data.get("session_id")
        return sid if isinstance(sid, str) and sid else None
    except (json.JSONDecodeError, OSError):
        return None


def latest_live_session_from_manifest() -> str | None:
    """Session that contains the most recently written live crop."""
    by_session: dict[str, float] = {}
    for e in read_manifest_entries():
        if e.source != SOURCE_LIVE or not e.session_id:
            continue
        crop = RESULT_CLS_CROPS_DIR / f"{e.stem}.jpg"
        if not crop.is_file():
            continue
        mtime = crop.stat().st_mtime
        by_session[e.session_id] = max(by_session.get(e.session_id, 0.0), mtime)
    if not by_session:
        return None
    return max(by_session, key=by_session.get)


def resolve_training_live_session_id(active_session_id: str | None = None) -> str | None:
    """Current live session, else last persisted, else newest live session in manifest."""
    if active_session_id:
        return active_session_id
    persisted = load_persisted_live_session_id()
    if persisted:
        return persisted
    return latest_live_session_from_manifest()


def count_scoped_live_stems(session_id: str | None, scope: str) -> int:
    entries = read_manifest_entries()
    filtered = filter_entries_by_scope(entries, scope, session_id)
    return sum(1 for e in filtered if (RESULT_CLS_CROPS_DIR / f"{e.stem}.jpg").is_file())


# ─── Label parsing ────────────────────────────────────────────────────────────

def _parse_label_to_quads(label_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    quads: list[np.ndarray] = []
    if not label_path.is_file():
        return quads
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 8:
            pts = np.array(
                [[vals[i] * img_w, vals[i + 1] * img_h] for i in range(0, 8, 2)],
                dtype=np.float32,
            )
            quads.append(pts)
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            pts = np.array(
                [[vals[i] * img_w, vals[i + 1] * img_h] for i in range(0, len(vals), 2)],
                dtype=np.float32,
            )
            if len(pts) == 4:
                quads.append(pts)
    return quads


def _cv_imread(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def export_crop_for_image(
    image_path: Path,
    label_path: Path,
    output_dir: Path | None = None,
) -> Path | None:
    output_dir = output_dir or RESULT_CLS_CROPS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    bgr = _cv_imread(image_path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    quads = _parse_label_to_quads(label_path, w, h)
    if not quads:
        return None
    quad = order_quad_points(quads[0])
    try:
        crop = warp_quad_crop(bgr, quad)
    except Exception:
        return None
    crop_lb = letterbox(crop, RESULT_CLS_IMGSZ)
    out_path = output_dir / f"{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), crop_lb)
    return out_path


def export_crops_for_folder(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    output_dir = output_dir or RESULT_CLS_CROPS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    total = len(image_files)
    for i, img_path in enumerate(image_files):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            results[stem] = "no_label"
        else:
            out = export_crop_for_image(img_path, label_path, output_dir)
            results[stem] = "ready" if out else "warp_failed"
        if progress_cb:
            progress_cb(i + 1, total, stem)
    return results


# ─── Augmentation ─────────────────────────────────────────────────────────────

def _stem_augment_seed(stem: str, aug_index: int) -> int:
    return (hash(stem) ^ (aug_index * 10007)) & 0x7FFFFFFF


def augment_result_crop(bgr: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = bgr.astype(np.float32)
    alpha = float(rng.uniform(0.9, 1.1))
    beta = float(rng.uniform(-10, 10))
    img = np.clip(img * alpha + beta, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-6, 6)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.75, 1.25), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.80, 1.20), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.rand() > 0.60:
        img += rng.normal(0, float(rng.uniform(2, 4)), img.shape).astype(np.float32)
    if rng.rand() > 0.85:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img.astype(np.uint8), (k, k), 0).astype(np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    angle = float(rng.uniform(-4, 4))
    tx = float(rng.uniform(-0.04, 0.04) * w)
    ty = float(rng.uniform(-0.04, 0.04) * h)
    scale = float(rng.uniform(0.94, 1.06))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(
        out, m, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    if rng.rand() > 0.7:
        quality = int(rng.randint(65, 95))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return letterbox(out, RESULT_CLS_IMGSZ)


def _clear_stem_augmented_files(stem: str, cls: str, augmented_dir: Path) -> None:
    class_dir = augmented_dir / cls
    if not class_dir.is_dir():
        return
    for p in class_dir.glob(f"{stem}_aug*.jpg"):
        try:
            p.unlink()
        except OSError:
            pass


def _augmented_paths_for_stem(stem: str, cls: str, augmented_dir: Path) -> list[Path]:
    class_dir = augmented_dir / cls
    if not class_dir.is_dir():
        return []
    return sorted(class_dir.glob(f"{stem}_aug*.jpg"))


def generate_augmented_for_stem(
    stem: str,
    cls: str,
    crops_dir: Path | None = None,
    augmented_dir: Path | None = None,
    per_stem: int | None = None,
) -> int:
    crops_dir = crops_dir or RESULT_CLS_CROPS_DIR
    augmented_dir = augmented_dir or RESULT_CLS_AUGMENTED_DIR
    per_stem = per_stem if per_stem is not None else RESULT_CLS_AUG_PER_STEM
    (augmented_dir / cls).mkdir(parents=True, exist_ok=True)
    src = crops_dir / f"{stem}.jpg"
    if not src.is_file():
        return 0
    bgr = _cv_imread(src)
    if bgr is None:
        return 0
    _clear_stem_augmented_files(stem, cls, augmented_dir)
    for j in range(per_stem):
        seed = _stem_augment_seed(stem, j)
        aug_bgr = augment_result_crop(bgr, seed)
        out_path = augmented_dir / cls / f"{stem}_aug{j:02d}.jpg"
        cv2.imwrite(str(out_path), aug_bgr)
    return per_stem


def generate_augmented_crops(
    manifest: dict[str, str],
    crops_dir: Path | None = None,
    augmented_dir: Path | None = None,
    per_stem: int | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    crops_dir = crops_dir or RESULT_CLS_CROPS_DIR
    augmented_dir = augmented_dir or RESULT_CLS_AUGMENTED_DIR
    per_stem = per_stem if per_stem is not None else RESULT_CLS_AUG_PER_STEM
    if per_stem < 1:
        raise ValueError("per_stem must be at least 1")
    for cls in RESULT_CLASSES:
        (augmented_dir / cls).mkdir(parents=True, exist_ok=True)
    labeled = [(s, manifest[s]) for s in sorted(manifest.keys()) if manifest[s] in RESULT_CLASSES]
    total = len(labeled)
    written = 0
    skipped = 0
    index: dict[str, dict] = {}
    for i, (stem, cls) in enumerate(labeled):
        n = generate_augmented_for_stem(stem, cls, crops_dir, augmented_dir, per_stem)
        if n == 0:
            skipped += 1
        else:
            written += n
            for j in range(per_stem):
                index[f"{stem}_aug{j:02d}.jpg"] = {
                    "stem": stem, "class": cls, "aug_index": j, "seed": _stem_augment_seed(stem, j),
                }
        if progress_cb:
            progress_cb(i + 1, total, stem)
    index_path = RESULT_CLS_AUGMENT_INDEX
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {
        "stems_processed": total - skipped,
        "stems_skipped": skipped,
        "images_written": written,
        "per_stem": per_stem,
        "augmented_dir": str(augmented_dir),
    }


def _append_live_correction_log(
    stem: str,
    correct_class: str,
    predicted_class: str | None,
    session_id: str,
) -> None:
    RESULT_CLS_LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "stem": stem,
        "correct": correct_class,
        "predicted": predicted_class or "",
        "session_id": session_id,
    }
    with open(RESULT_CLS_LIVE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_live_correction(
    crop_bgr: np.ndarray,
    correct_class: str,
    poi_index: int,
    session_id: str,
    predicted_class: str | None = None,
    per_stem: int | None = None,
    auto_augment: bool = True,
) -> dict:
    if correct_class not in RESULT_CLASSES:
        raise ValueError(f"Invalid class: {correct_class}")
    stem = f"live_{int(time.time() * 1000)}_p{poi_index}"
    crop_lb = letterbox(crop_bgr, RESULT_CLS_IMGSZ)
    RESULT_CLS_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(RESULT_CLS_CROPS_DIR / f"{stem}.jpg"), crop_lb)
    upsert_manifest_entry(
        ManifestEntry(stem=stem, cls=correct_class, source=SOURCE_LIVE, session_id=session_id),
    )
    persist_live_session_id(session_id)
    n_aug = 0
    if auto_augment:
        n_aug = generate_augmented_for_stem(stem, correct_class, per_stem=per_stem)
    _append_live_correction_log(stem, correct_class, predicted_class, session_id)
    return {
        "stem": stem,
        "class": correct_class,
        "n_aug": n_aug,
        "session_id": session_id,
        "predicted": predicted_class,
    }


# ─── Bundle builder ───────────────────────────────────────────────────────────

def build_result_cls_bundle(
    manifest: dict[str, str] | None = None,
    entries: list[ManifestEntry] | None = None,
    crops_dir: Path | None = None,
    bundle_dir: Path | None = None,
    augmented_dir: Path | None = None,
    val_fraction: float = _VAL_FRACTION,
    include_augmented: bool = True,
    training_scope: str = SCOPE_FULL,
    live_session_id: str | None = None,
) -> tuple[Path, int, int, dict]:
    if entries is None:
        if manifest is None:
            entries = read_manifest_entries()
        else:
            entries = [
                ManifestEntry(stem=s, cls=c, source=SOURCE_LABELED, session_id="legacy")
                for s, c in manifest.items()
                if c in RESULT_CLASSES
            ]
    entries = filter_entries_by_scope(entries, training_scope, live_session_id)

    crops_dir = crops_dir or RESULT_CLS_CROPS_DIR
    bundle_dir = bundle_dir or RESULT_CLS_BUNDLE_DIR
    augmented_dir = augmented_dir or RESULT_CLS_AUGMENTED_DIR

    for split in ("train", "val"):
        for cls in RESULT_CLASSES:
            dir_path = bundle_dir / split / cls
            if dir_path.is_dir():
                for f in dir_path.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                        except OSError:
                            pass
            else:
                dir_path.mkdir(parents=True, exist_ok=True)

    stems_ok = sorted(
        (e for e in entries if (crops_dir / f"{e.stem}.jpg").is_file()),
        key=lambda e: e.stem,
    )
    rng = random.Random(_BUNDLE_SEED)
    rng.shuffle(stems_ok)
    if len(stems_ok) <= 1:
        n_val = 0
    else:
        n_val = max(1, int(len(stems_ok) * val_fraction))
        n_val = min(n_val, len(stems_ok) - 1)
    val_stems = {e.stem for e in stems_ok[:n_val]}

    n_train, n_val_actual = 0, 0
    n_aug_train, n_aug_val = 0, 0

    for entry in stems_ok:
        stem, cls = entry.stem, entry.cls
        split = "val" if stem in val_stems else "train"
        dest_root = bundle_dir / split / cls
        shutil.copy2(crops_dir / f"{stem}.jpg", dest_root / f"{stem}.jpg")
        if split == "train":
            n_train += 1
        else:
            n_val_actual += 1
        if include_augmented and augmented_dir.is_dir():
            for aug_path in _augmented_paths_for_stem(stem, cls, augmented_dir):
                shutil.copy2(aug_path, dest_root / aug_path.name)
                if split == "train":
                    n_aug_train += 1
                else:
                    n_aug_val += 1

    stats = {
        "n_orig_train": n_train,
        "n_orig_val": n_val_actual,
        "n_aug_train": n_aug_train,
        "n_aug_val": n_aug_val,
        "n_train_total": n_train + n_aug_train,
        "n_val_total": n_val_actual + n_aug_val,
        "n_stems": len(stems_ok),
        "training_scope": training_scope,
        "live_session_id": live_session_id or "",
    }
    return bundle_dir, n_train + n_aug_train, n_val_actual + n_aug_val, stats


# ─── Training ─────────────────────────────────────────────────────────────────

def train_resnet34(
    bundle_dir: Path | None = None,
    epochs: int = 100,
    batch: int = 32,
    lr: float = 1e-4,
    patience: int = 20,
    device: str | None = None,
    progress_cb: Callable[[int, int, float, float], None] | None = None,
    resume_from: Path | None = None,
) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    from studio.config import resolve_result_cls_model_path

    bundle_dir = bundle_dir or RESULT_CLS_BUNDLE_DIR
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(bundle_dir / "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(str(bundle_dir / "val"), transform=val_transform)

    if len(train_ds) == 0:
        raise RuntimeError("Training set is empty. Label more images or widen training scope.")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    resumed = False
    if resume_from is None:
        resume_from = resolve_result_cls_model_path()
    if resume_from is not None and resume_from.is_file():
        ckpt = torch.load(str(resume_from), map_location=device, weights_only=False)
        arch = ckpt.get("arch", RESULT_CLS_ARCH)
        if arch == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(RESULT_CLASSES))
        else:
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(RESULT_CLASSES))
        model.load_state_dict(ckpt["state_dict"])
        resumed = True
    else:
        if RESULT_CLS_ARCH == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, len(RESULT_CLASSES))
        else:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(RESULT_CLASSES))
    model = model.to(device)

    targets = np.array(train_ds.targets)
    class_counts = [np.sum(targets == i) for i in range(len(RESULT_CLASSES))]
    total_samples = len(targets)
    class_weights = [total_samples / max(1, c) for c in class_counts]
    weight_sum = sum(class_weights)
    class_weights = [w * len(RESULT_CLASSES) / weight_sum for w in class_weights]
    weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_per_class_recall: dict = {}
    best_epoch = 0
    wait = 0

    out_dir = RESULT_CLS_RUNS_DIR / RESULT_CLS_ARCH
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    class_names = train_ds.classes

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        per_class_correct = [0] * len(RESULT_CLASSES)
        per_class_total = [0] * len(RESULT_CLASSES)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                for c in range(len(RESULT_CLASSES)):
                    mask = labels == c
                    per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    per_class_total[c] += mask.sum().item()

        val_loss = val_loss_sum / max(len(val_ds), 1)
        val_acc = correct / max(total, 1)
        current_recall = {}
        for i, cls_name in enumerate(class_names):
            if per_class_total[i] > 0:
                current_recall[cls_name] = per_class_correct[i] / per_class_total[i]
            else:
                current_recall[cls_name] = 0.0

        if progress_cb:
            progress_cb(epoch + 1, epochs, train_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_per_class_recall = current_recall.copy()
            best_epoch = epoch + 1
            wait = 0
            torch.save({
                "arch": RESULT_CLS_ARCH,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "imgsz": 448,
                "epoch": epoch + 1,
            }, best_path)
        else:
            wait += 1
            if wait >= patience:
                break

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_acc": best_val_acc,
        "per_class_recall": best_per_class_recall,
        "class_names": class_names,
        "total_epochs_run": epoch + 1,
        "resumed_from_checkpoint": resumed,
    }

    meta = {
        "arch": RESULT_CLS_ARCH,
        "class_names": class_names,
        "imgsz": 448,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ─── Inference ────────────────────────────────────────────────────────────────

class ResNet34Classifier:
    """Load a trained result-classifier checkpoint and predict on BGR crops."""

    def __init__(self, checkpoint_path: Path, device: str | None = None):
        import torch
        from torchvision import models
        import torch.nn as nn

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        self.class_names: list[str] = ckpt.get("class_names", list(RESULT_CLASSES))
        self.imgsz: int = ckpt.get("imgsz", RESULT_CLS_IMGSZ)

        arch = ckpt.get("arch", RESULT_CLS_ARCH)
        if arch == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        else:
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()
        self._model = model

        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.imgsz, self.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def predict(self, bgr_crop: np.ndarray) -> tuple[str, float]:
        torch = self._torch
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return self.class_names[idx.item()], conf.item()


def load_result_classifier(path: Path | None = None, device: str | None = None) -> ResNet34Classifier | None:
    from studio.config import resolve_result_cls_model_path
    if path is None:
        path = resolve_result_cls_model_path()
    if path is None or not path.is_file():
        return None
    return ResNet34Classifier(path, device=device)
