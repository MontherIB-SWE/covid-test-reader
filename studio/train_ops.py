"""Training dataset bundle preparation (no UI). Used by studio.qt.train_tab."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from studio.config import (
    OUTPUTS_DIR,
    TRAIN_BUNDLE_DIR,
    TRAIN_DATA_YAML,
    SUPPORTED_EXTENSIONS,
)

_VAL_FRACTION = 0.15
_BUNDLE_SEED = 42


def train_device() -> int | str:
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _collect_pairs_for_source(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    label_by_stem: dict[str, Path] = {}
    for lab_path in labels_dir.rglob("*.txt"):
        if lab_path.is_file():
            label_by_stem[lab_path.stem.lower()] = lab_path
    pairs: list[tuple[Path, Path]] = []
    for img_path in images_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        lab = label_by_stem.get(img_path.stem.lower())
        if lab is not None:
            pairs.append((img_path, lab))
    pairs.sort(key=lambda t: str(t[0]).lower())
    return pairs


def _label_file_has_objects(lab_path: Path) -> bool:
    try:
        text = lab_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return False
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            return True
    return False


def _standard_images_dir(root: Path) -> Path | None:
    d = root / "images"
    return d if d.is_dir() else None


def _standard_labels_dir(root: Path) -> Path | None:
    d = root / "labels"
    return d if d.is_dir() else None


def _collect_positive_pairs_tagged(root: Path | None, source_tag: int) -> list[tuple[Path, Path, int]]:
    if root is None or not root.is_dir():
        return []
    img_d = _standard_images_dir(root)
    lbl_d = _standard_labels_dir(root)
    if img_d is None or lbl_d is None:
        return []
    out: list[tuple[Path, Path, int]] = []
    for img_p, lab_p in _collect_pairs_for_source(img_d, lbl_d):
        if _label_file_has_objects(lab_p):
            out.append((img_p, lab_p, source_tag))
    return out


def _enumerate_negative_images(neg_root: Path | None, labeled_paths: set[str]) -> list[Path]:
    if neg_root is None or not neg_root.is_dir():
        return []
    img_root = _standard_images_dir(neg_root)
    if img_root is None:
        img_root = neg_root
    found: list[Path] = []
    for p in img_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        key = str(p.resolve())
        if key in labeled_paths:
            continue
        found.append(p)
    found.sort(key=lambda x: str(x).lower())
    return found


def _split_train_val_list(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    if not items:
        return [], []
    rnd = random.Random(seed)
    sh = list(items)
    rnd.shuffle(sh)
    if len(sh) == 1:
        return sh, []
    n_val = max(1, int(round(len(sh) * val_fraction)))
    n_val = min(n_val, len(sh) - 1)
    return sh[n_val:], sh[:n_val]


BundleRow = tuple[Path, Path | None, int]


def _reset_bundle_dirs() -> tuple[Path, Path, Path, Path]:
    if TRAIN_BUNDLE_DIR.exists():
        shutil.rmtree(TRAIN_BUNDLE_DIR, ignore_errors=True)
    train_img = TRAIN_BUNDLE_DIR / "images" / "train"
    train_lbl = TRAIN_BUNDLE_DIR / "labels" / "train"
    val_img = TRAIN_BUNDLE_DIR / "images" / "val"
    val_lbl = TRAIN_BUNDLE_DIR / "labels" / "val"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)
    return train_img, train_lbl, val_img, val_lbl


def _copy_bundle_rows(rows: list[BundleRow], img_dir: Path, lbl_dir: Path) -> None:
    for idx, (img_p, lab_p, si) in enumerate(rows):
        stem = f"src{si:02d}_{idx:05d}_{img_p.stem}"[:180]
        dst_i = img_dir / f"{stem}{img_p.suffix}"
        dst_l = lbl_dir / f"{stem}.txt"
        shutil.copy2(img_p, dst_i)
        if lab_p is None:
            dst_l.write_text("", encoding="utf-8")
        else:
            shutil.copy2(lab_p, dst_l)


def _write_data_yaml() -> None:
    root_abs = TRAIN_BUNDLE_DIR.resolve()
    yaml_text = (
        f"path: {root_abs.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: poi\n"
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_YAML.write_text(yaml_text, encoding="utf-8")


def prepare_mixed_train_bundle(
    generated_root: Path | None,
    labeled_poi_root: Path | None,
    negatives_root: Path | None,
    *,
    labeled_fraction: float,
    val_fraction: float = _VAL_FRACTION,
    seed: int = _BUNDLE_SEED,
) -> tuple[Path, int, int, str]:
    if not (0 < labeled_fraction <= 1.0):
        raise ValueError("Labeled fraction must be between 0 and 100% (e.g. 20 for 20% POI images).")

    tagged: list[tuple[Path, Path, int]] = []
    tagged.extend(_collect_positive_pairs_tagged(generated_root, 0))
    tagged.extend(_collect_positive_pairs_tagged(labeled_poi_root, 1))

    by_res: dict[str, tuple[Path, Path, int]] = {}
    for img_p, lab_p, tag in tagged:
        k = str(img_p.resolve())
        if k not in by_res:
            by_res[k] = (img_p, lab_p, tag)
    labeled_rows: list[tuple[Path, Path, int]] = list(by_res.values())

    if len(labeled_rows) < 1:
        raise ValueError(
            "No images with non-empty POI labels were found under generated / labeled folders "
            "(need …/images + …/labels with matching names).",
        )

    labeled_paths = {str(t[0].resolve()) for t in labeled_rows}
    neg_root_use = negatives_root if (negatives_root and negatives_root.is_dir()) else None
    neg_pool = _enumerate_negative_images(neg_root_use, labeled_paths)

    n_l = len(labeled_rows)
    target_total = int(round(n_l / labeled_fraction))
    n_neg_needed = max(0, target_total - n_l)

    rnd = random.Random(seed + 7)
    neg_pool_shuffled = list(neg_pool)
    rnd.shuffle(neg_pool_shuffled)
    neg_chosen = neg_pool_shuffled[: min(n_neg_needed, len(neg_pool_shuffled))]

    actual_total = n_l + len(neg_chosen)
    actual_labeled_frac = n_l / actual_total if actual_total else 0.0

    train_l, val_l = _split_train_val_list(labeled_rows, val_fraction, seed)
    neg_as_rows: list[tuple[Path, None, int]] = [(p, None, 2) for p in neg_chosen]
    train_n, val_n = _split_train_val_list(neg_as_rows, val_fraction, seed + 1)

    train_bundle: list[BundleRow] = [(a, b, c) for a, b, c in train_l] + [(a, b, c) for a, b, c in train_n]
    val_bundle: list[BundleRow] = [(a, b, c) for a, b, c in val_l] + [(a, b, c) for a, b, c in val_n]

    if len(train_bundle) < 1:
        raise ValueError("Train split ended up empty — add more data.")
    if not val_bundle:
        if len(train_bundle) < 2:
            raise ValueError(
                "After mixing, fewer than 2 images are available for train/val — add more labeled or negative images.",
            )
        val_bundle.append(train_bundle.pop())

    train_img, train_lbl, val_img, val_lbl = _reset_bundle_dirs()
    _copy_bundle_rows(train_bundle, train_img, train_lbl)
    _copy_bundle_rows(val_bundle, val_img, val_lbl)
    _write_data_yaml()

    summary = (
        f"{n_l} POI images (all used), {len(neg_chosen)} negatives "
        f"(wanted {n_neg_needed}, pool had {len(neg_pool)}). "
        f"Mix ≈ {actual_labeled_frac * 100:.1f}% labeled. "
        f"Train {len(train_bundle)} / val {len(val_bundle)}."
    )
    if len(neg_chosen) < n_neg_needed:
        summary += " Not enough negatives for exact ratio — used all available."

    return TRAIN_DATA_YAML.resolve(), len(train_bundle), len(val_bundle), summary


def prepare_train_bundle(
    sources: list[tuple[Path, Path]],
    *,
    val_fraction: float = _VAL_FRACTION,
    seed: int = _BUNDLE_SEED,
) -> tuple[Path, int, int]:
    all_rows: list[tuple[Path, Path, int]] = []
    for si, (img_d, lbl_d) in enumerate(sources):
        for img_p, lab_p in _collect_pairs_for_source(img_d, lbl_d):
            all_rows.append((img_p, lab_p, si))

    if not all_rows:
        raise ValueError("No image + label pairs found across your folders.")
    if len(all_rows) < 2:
        raise ValueError(
            "Need at least 2 labeled images total (across all folders) to split train / validation."
        )

    rnd = random.Random(seed)
    rnd.shuffle(all_rows)

    n = len(all_rows)
    n_val = max(1, int(round(n * val_fraction)))
    if n <= 3:
        n_val = 1
    n_val = min(n_val, n - 1)

    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    train_img, train_lbl, val_img, val_lbl = _reset_bundle_dirs()
    _copy_bundle_rows(train_rows, train_img, train_lbl)
    _copy_bundle_rows(val_rows, val_img, val_lbl)
    _write_data_yaml()

    return TRAIN_DATA_YAML.resolve(), len(train_rows), len(val_rows)


VAL_FRACTION = _VAL_FRACTION
