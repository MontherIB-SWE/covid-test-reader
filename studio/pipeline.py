"""Dataset pipeline: extract assets, synthetic generation, reorganize."""
from __future__ import annotations

from typing import Callable

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from studio.config import (
    DATA_ASSETS_DIR,
    DATA_GENERATED_DIR,
    DATA_LABELED_DIR,
    DATA_NEGATIVES_DIR,
)

# ===== MERGED FROM extract_assets.py =====

"""Extract POI assets from labeled images to create transparent PNGs.

Saves each POI as a cropped BGRA image with an eroded alpha mask, plus a
sidecar .txt with 4 corner coordinates (pixel-space, relative to crop).
"""

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ERODE_PX = 3
PAD = 12  # padding around the polygon bounding box in the crop


def _cv_imread_unicode(path: Path) -> np.ndarray | None:
    p = str(path)
    img = cv2.imread(p)
    if img is not None:
        return img
    try:
        buf = np.fromfile(p, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _cv_imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    p = str(path)
    ext = path.suffix.lower() or ".png"
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        enc.tofile(p)
        return True
    except Exception:
        return cv2.imwrite(p, img)


def _label_line_to_polygon_px(parts: list[str], w: int, h: int) -> np.ndarray | None:
    """YOLO OBB (class + 8 floats) or polygon (class + pairs of normalized coords)."""
    if len(parts) < 2:
        return None
    try:
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    n = len(vals)
    if n == 8:
        return np.array(
            [
                [vals[0] * w, vals[1] * h],
                [vals[2] * w, vals[3] * h],
                [vals[4] * w, vals[5] * h],
                [vals[6] * w, vals[7] * h],
            ],
            dtype=np.float32,
        )
    if n >= 6 and n % 2 == 0:
        poly = np.array(
            [[vals[i] * w, vals[i + 1] * h] for i in range(0, n, 2)],
            dtype=np.float32,
        )
        if len(poly) >= 3:
            return poly
    return None


def extract_assets(
    images_dirs: list[Path] | None = None,
    labels_dirs: list[Path] | None = None,
    output_dir: Path = None,  # defaults to DATA_ASSETS_DIR
) -> dict:
    """Extract POI assets from labeled images.

    images_dirs: list of image folders (or a single folder).
    labels_dirs: matching list of label folders.  If None, each image folder
                 is assumed to have a sibling ``labels/`` or ``../labels/``.
    output_dir:  where to write cropped PNG assets.

    Returns dict with counts.
    """
    if images_dirs is None:
        images_dirs = [DATA_LABELED_DIR / "images"]
    if isinstance(images_dirs, (str, Path)):
        images_dirs = [Path(images_dirs)]
    images_dirs = [Path(p) for p in images_dirs]

    if labels_dirs is None:
        labels_dirs = []
        for img_d in images_dirs:
            # Try sibling labels/ first, then ../labels/
            candidate = img_d.parent / "labels"
            if not candidate.is_dir():
                candidate = img_d / ".." / "labels"
            labels_dirs.append(candidate)
    elif isinstance(labels_dirs, (str, Path)):
        labels_dirs = [Path(labels_dirs)]
    labels_dirs = [Path(p) for p in labels_dirs]

    if output_dir is None:
        output_dir = DATA_ASSETS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0
    corner_oob_count = 0

    for images_dir, labels_dir in zip(images_dirs, labels_dirs):
        if not labels_dir.exists() or not images_dir.exists():
            print(f"Skip: {labels_dir} or {images_dir} not found")
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"Found {len(label_files)} label files in {labels_dir}. Extracting assets...")

        for lbl_path in label_files:
            stem = lbl_path.stem
            img_path = None
            for ext in EXTS:
                p = images_dir / f"{stem}{ext}"
                if p.is_file():
                    img_path = p
                    break
            if img_path is None:
                for ext in EXTS:
                    hits = list(images_dir.rglob(f"{stem}{ext}"))
                    if hits:
                        img_path = hits[0]
                        break

            if img_path is None:
                skipped_count += 1
                continue

            img = _cv_imread_unicode(img_path)
            if img is None:
                skipped_count += 1
                continue

            h, w = img.shape[:2]

            try:
                raw = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                skipped_count += 1
                continue
            lines = raw.split("\n") if raw else []

            poly_idx = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                pts_f = _label_line_to_polygon_px(parts, w, h)
                if pts_f is None:
                    continue
                pts = pts_f.astype(np.int32)
                if len(pts) < 3:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                kernel = np.ones((ERODE_PX * 2 + 1, ERODE_PX * 2 + 1), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)

                if mask.max() == 0:
                    skipped_count += 1
                    continue

                x, y, bw, bh = cv2.boundingRect(pts)
                x1 = max(0, x - PAD)
                y1 = max(0, y - PAD)
                x2 = min(w, x + bw + PAD)
                y2 = min(h, y + bh + PAD)

                crop_img = img[y1:y2, x1:x2]
                crop_mask = mask[y1:y2, x1:x2]

                crop_bgra = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
                crop_bgra[:, :, 3] = crop_mask

                _fill_transparent_bgr(crop_bgra)

                asset_name = f"{stem}_poi_{poly_idx}.png"
                out_path = output_dir / asset_name
                if not _cv_imwrite_unicode(out_path, crop_bgra):
                    skipped_count += 1
                    continue

                txt_name = f"{stem}_poi_{poly_idx}.txt"
                txt_path = output_dir / txt_name
                pts_crop = pts_f - np.array([x1, y1], dtype=np.float32)
                crop_w, crop_h = x2 - x1, y2 - y1
                pts_crop[:, 0] = np.clip(pts_crop[:, 0], 0, crop_w)
                pts_crop[:, 1] = np.clip(pts_crop[:, 1], 0, crop_h)
                flat_pts = pts_crop.flatten()
                txt_path.write_text(" ".join(f"{v:.6f}" for v in flat_pts), encoding="utf-8")

                extracted_count += 1
                poly_idx += 1

    print("=" * 50)
    print(f"Extraction Complete!")
    print(f"Assets extracted: {extracted_count}")
    print(f"Skipped/Empty: {skipped_count}")
    print(f"Saved to: {output_dir.resolve()}")
    print("=" * 50)
    return {
        "extracted": extracted_count,
        "skipped": skipped_count,
        "corner_oob": corner_oob_count,
    }


def _fill_transparent_bgr(bgra: np.ndarray) -> None:
    """Replace BGR of fully-transparent pixels with nearest opaque neighbour.

    After erosion, pixels just outside the eroded mask have alpha=0 but still
    carry original image colours (or black from the eroded boundary).  When
    the synthetic generator later applies GaussianBlur to the alpha channel,
    these pixels bleed into the visible edge.  If their BGR is black (0,0,0)
    the result is a dark fringe around the pasted POI.

    Filling them with the nearest opaque colour ensures the blur produces a
    smooth colour transition instead of a dark halo.
    """
    alpha = bgra[:, :, 3]
    if alpha.max() == 0:
        return  # entirely transparent – nothing to do

    # Dilate the opaque BGR region by 1px using a 3×3 nearest-neighbour
    # flood.  This is fast and covers the immediate border pixels that the
    # generator's 7×7 GaussianBlur will reach.
    opaque_mask = (alpha > 0).astype(np.uint8)
    for c in range(3):
        channel = bgra[:, :, c].astype(np.float32)
        # Average of opaque neighbours
        total = channel.copy()
        count = opaque_mask.astype(np.float32)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_mask = np.roll(opaque_mask, (dy, dx), axis=(0, 1))
            shifted_ch = np.roll(channel, (dy, dx), axis=(0, 1))
            valid = shifted_mask > 0
            total += shifted_ch * valid
            count += valid.astype(np.float32)

        transparent = alpha == 0
        has_opaque_neighbour = transparent & (count > 0)
        bgra[has_opaque_neighbour, c] = np.clip(
            total[has_opaque_neighbour] / count[has_opaque_neighbour], 0, 255
        ).astype(np.uint8)


# ===== MERGED FROM generate_synthetic.py =====

"""Generate synthetic images by pasting transparent POI assets onto negative backgrounds.

Each output image may contain 1-3 POI instances pasted at random positions,
with random scale, rotation, and colour jitter.  Labels are in YOLO OBB
format (class x1 y1 x2 y2 x3 y3 x4 y4, normalised).
"""

# Configuration
BG_DIR = DATA_NEGATIVES_DIR / "images"
ASSETS_DIR = DATA_ASSETS_DIR
OUT_IMG_DIR = DATA_GENERATED_DIR / "images"
OUT_LBL_DIR = DATA_GENERATED_DIR / "labels"

TARGET_SIZE = 960

# Probability weights for number of POIs per image  (1, 2, 3)
MULTI_POI_WEIGHTS = [0.60, 0.30, 0.10]

# Placement: axis-aligned bbox clearance between pasted POIs (pixels); retries before skipping.
POI_BBOX_GAP_PX = 10
POI_PLACEMENT_TRIES = 72
POI_SLOT_ASSET_TRIES = 48  # per POI slot: different assets / random layouts


def _get_files(directory: Path) -> list[Path]:
    files = []
    if not directory.exists():
        return files
    for f in directory.iterdir():
        if f.suffix.lower() in EXTS:
            files.append(f)
    return files


def _resize_background(bg: np.ndarray, target: int) -> np.ndarray:
    """Resize background to target×target while preserving aspect ratio.

    Crops from the centre to fill the square exactly, avoiding any distortion.
    """
    h, w = bg.shape[:2]
    # Pick the largest centred square
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    cropped = bg[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


def rotate_image(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a BGRA image keeping full bounds.

    Transparent border pixels have their BGR set to the nearest opaque colour
    to prevent black fringes after later Gaussian blur on the alpha channel.

    Returns (rotated_image, rotation_matrix).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Kill black fringes: replace BGR of transparent pixels with the nearest
    # opaque neighbour so that the subsequent GaussianBlur on alpha does not
    # bleed black into the visible edge.
    _defringe_bgra(rotated)

    return rotated, M


def _defringe_bgra(bgra: np.ndarray) -> None:
    """Replace BGR of transparent pixels with nearest opaque colour.

    Uses two passes of neighbour averaging to cover the full 7×7 Gaussian
    kernel reach (~3 px from the opaque boundary).
    """
    alpha = bgra[:, :, 3]
    if alpha.max() == 0:
        return

    opaque = (alpha > 0).astype(np.float32)

    # Two passes: after pass 1, the fill extends 1 px into transparent area;
    # after pass 2 it extends 2 px (covering the ~3 px blur radius).
    for _ in range(2):
        transparent = (alpha == 0).astype(np.float32)
        for c in range(3):
            ch = bgra[:, :, c].astype(np.float32)
            total = ch * opaque
            count = opaque.copy()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                shifted_opaque = np.roll(opaque, (dy, dx), axis=(0, 1))
                shifted_ch = np.roll(ch, (dy, dx), axis=(0, 1))
                total += shifted_ch * shifted_opaque
                count += shifted_opaque
            needs_fill = (transparent > 0) & (count > 0)
            bgra[needs_fill, c] = np.clip(
                total[needs_fill] / count[needs_fill], 0, 255
            ).astype(np.uint8)
            # Mark newly filled pixels as opaque for the next pass
            opaque[needs_fill] = 1.0


def apply_color_jitter(image: np.ndarray, brightness: float = 0.2,
                       contrast: float = 0.2) -> np.ndarray:
    """Apply random brightness and contrast to BGR channels (ignoring alpha)."""
    alpha = image[:, :, 3:]
    bgr = image[:, :, :3].astype(np.float32)

    b_shift = random.uniform(-brightness, brightness) * 255
    bgr += b_shift

    c_mult = random.uniform(1.0 - contrast, 1.0 + contrast)
    bgr = (bgr - 128) * c_mult + 128

    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return np.concatenate([bgr, alpha], axis=2)


def _load_asset(asset_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a BGRA asset and its 4-corner sidecar.  Returns None on failure."""
    asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    if asset is None or asset.ndim != 3 or asset.shape[2] != 4:
        return None

    txt_path = asset_path.with_suffix('.txt')
    if not txt_path.exists():
        return None

    pts_str = txt_path.read_text().strip().split()
    if len(pts_str) != 8:
        return None

    corners = np.array([float(x) for x in pts_str], dtype=np.float32).reshape(4, 2)
    return asset, corners


def _place_one_poi(
    bg: np.ndarray,
    asset_path: Path,
    occupied_boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, str] | None:
    """Paste one POI onto *bg* and return (final_corners_in_image, label_line).

    Checks against *occupied_boxes* (x1,y1,x2,y2 in image px) to avoid
    excessive overlap with previously placed POIs.  Returns None if the
    asset can't be loaded or doesn't fit.
    """
    result = _load_asset(asset_path)
    if result is None:
        return None
    asset, corners = result

    # --- Scale ---
    max_dim = max(asset.shape[0], asset.shape[1])
    max_allowed = TARGET_SIZE * 0.5

    scale = random.uniform(0.5, 1.5)
    if (max_dim * scale) > max_allowed:
        scale = max_allowed / max_dim

    new_w = int(asset.shape[1] * scale)
    new_h = int(asset.shape[0] * scale)
    if new_w < 4 or new_h < 4:
        return None

    scale_x = new_w / asset.shape[1]
    scale_y = new_h / asset.shape[0]
    scaled_corners = corners.copy()
    scaled_corners[:, 0] *= scale_x
    scaled_corners[:, 1] *= scale_y

    asset = cv2.resize(asset, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- Colour jitter ---
    asset = apply_color_jitter(asset)

    # --- Rotation ---
    angle = random.uniform(-45, 45)
    asset_rot, M = rotate_image(asset, angle)

    ones = np.ones(shape=(4, 1))
    points_ones = np.hstack([scaled_corners, ones])
    rotated_corners = M.dot(points_ones.T).T  # (4, 2)

    ah, aw = asset_rot.shape[:2]

    if aw > TARGET_SIZE or ah > TARGET_SIZE:
        return None

    # --- Placement: random tries until bbox clears existing POIs (gap), else abort ---
    px, py = 0, 0
    placed = False
    max_x = max(0, TARGET_SIZE - aw)
    max_y = max(0, TARGET_SIZE - ah)
    for _ in range(POI_PLACEMENT_TRIES):
        px = random.randint(0, max_x)
        py = random.randint(0, max_y)
        candidate = (px, py, px + aw, py + ah)
        if _bbox_clear_of_occupied(candidate, occupied_boxes, POI_BBOX_GAP_PX):
            placed = True
            break

    if not placed:
        return None

    final_corners = rotated_corners + np.array([px, py], dtype=np.float32)

    # --- Alpha composite with feathered edges ---
    alpha_mask = asset_rot[:, :, 3].astype(np.float32)
    alpha_mask = cv2.GaussianBlur(alpha_mask, (7, 7), 0)
    alpha_f = alpha_mask / 255.0
    alpha_inv = 1.0 - alpha_f

    bg_roi = bg[py:py + ah, px:px + aw]
    for c in range(3):
        bg_roi[:, :, c] = np.clip(
            alpha_f * asset_rot[:, :, c] + alpha_inv * bg_roi[:, :, c],
            0, 255,
        ).astype(np.uint8)
    bg[py:py + ah, px:px + aw] = bg_roi

    occupied_boxes.append((px, py, px + aw, py + ah))

    # --- Build label line ---
    norm = final_corners / TARGET_SIZE
    flat = norm.flatten()
    label_line = "0 " + " ".join(f"{v:.6f}" for v in flat)
    return bg, label_line


def _axis_aligned_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _bbox_clear_of_occupied(
    candidate: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
    gap_px: int,
) -> bool:
    """True if *candidate* does not intersect any occupied bbox inflated by *gap_px*."""
    for ox1, oy1, ox2, oy2 in occupied:
        inflated = (ox1 - gap_px, oy1 - gap_px, ox2 + gap_px, oy2 + gap_px)
        if _axis_aligned_intersects(candidate, inflated):
            return False
    return True


def _find_start_index(out_dir: Path | None = None) -> int:
    """Find the next available index by scanning existing output files."""
    _dir = out_dir if out_dir is not None else OUT_IMG_DIR
    if not _dir.exists():
        return 0
    max_idx = -1
    for f in _dir.iterdir():
        if f.stem.startswith("synth_"):
            try:
                idx = int(f.stem.split("_", 1)[1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


def generate(
    num_images: int = 10,
    on_saved: Callable[[Path], None] | None = None,
    bg_dir: Path | None = None,
    assets_dir: Path | None = None,
    out_img_dir: Path | None = None,
    out_lbl_dir: Path | None = None,
) -> None:
    _bg_dir = bg_dir if bg_dir is not None else BG_DIR
    _assets_dir = assets_dir if assets_dir is not None else ASSETS_DIR
    _out_img_dir = out_img_dir if out_img_dir is not None else OUT_IMG_DIR
    _out_lbl_dir = out_lbl_dir if out_lbl_dir is not None else OUT_LBL_DIR
    _out_img_dir.mkdir(parents=True, exist_ok=True)
    _out_lbl_dir.mkdir(parents=True, exist_ok=True)

    bg_files = _get_files(_bg_dir)
    asset_files = _get_files(_assets_dir)

    if not bg_files:
        print(f"Error: No backgrounds found in {_bg_dir}")
        return
    if not asset_files:
        print(f"Error: No POI assets found in {_assets_dir}")
        return

    start_idx = _find_start_index(_out_img_dir)
    print(f"Loaded {len(bg_files)} backgrounds and {len(asset_files)} POI assets.")
    print(f"Starting from index {start_idx}, generating {num_images} synthetic images...")

    for i in range(num_images):
        idx = start_idx + i
        bg_path = random.choice(bg_files)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue

        bg = _resize_background(bg, TARGET_SIZE)

        # Decide how many POIs to place
        num_pois = random.choices([1, 2, 3], weights=MULTI_POI_WEIGHTS, k=1)[0]

        label_lines: list[str] = []
        occupied: list[tuple[int, int, int, int]] = []

        for _ in range(num_pois):
            placed_slot = False
            for _attempt in range(POI_SLOT_ASSET_TRIES):
                asset_path = random.choice(asset_files)
                result = _place_one_poi(bg, asset_path, occupied)
                if result is not None:
                    _, label_line = result
                    label_lines.append(label_line)
                    placed_slot = True
                    break
            if not placed_slot:
                continue

        if not label_lines:
            continue  # skip images where nothing could be placed

        # Save image
        out_name = f"synth_{idx:05d}"
        out_img_path = _out_img_dir / f"{out_name}.jpg"
        cv2.imwrite(str(out_img_path), bg)
        if on_saved is not None:
            on_saved(out_img_path)

        # Save label (one line per POI)
        label_text = "\n".join(label_lines) + "\n"
        (_out_lbl_dir / f"{out_name}.txt").write_text(label_text)

        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print(f"Generated {i + 1}/{num_images} images (index {start_idx}–{idx})")

    print("\nGeneration Complete!")
    print(f"Output saved to: {_out_img_dir.resolve()}")


# ===== MERGED FROM reorganize_data.py =====

"""One-shot script to reorganize the dataset into a clean structure.

BEFORE:
  data/raw/Positive/          148 images (have POI)
  data/raw/Positive1/         300 images (have POI)
  data/raw/BadTests/            5 images (have POI)
  data/raw/outside/            35 images (have POI)
  data/raw/Negative/          272 images (have POI — misleading name!)
  data/raw/Negative1/          25 images (some have POI)
  data/raw/Negative2/         686 images (no POI, empty labels)
  data/raw/screenshare images/ 36 images + labels (have POI)
  data/raw/screenshare images no poi/  26 images + empty labels
  data/autolabel/             126 images + labels (have POI)
  data/labels_poly/          1471 labels (polygon format, mirrors raw/)
  data/labels/                360 labels (older bbox format)
  data/labels_poly_smooth/    460 labels (smoothed subset)
  data/test data/            1372 images (no labels)

AFTER:
  data/
  ├── images/
  │   ├── train/       ~1327 images  (80% split)
  │   ├── val/         ~166  images  (10% split)
  │   └── test/        ~166  images  (10% split)
  ├── labels/
  │   ├── train/       matching labels
  │   ├── val/
  │   └── test/
  └── test_data/
      ├── with_poi/    686 images (raw test data)
      └── without_poi/ 686 images (raw test data)

All training images are renamed with a source prefix to avoid collisions.
Old directories are moved to data/_archive/ (not deleted).

Usage:
    python reorganize_data.py          # preview only (dry run)
    python reorganize_data.py --run    # actually move files
"""

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
SEED = 42

# ── Source definitions ──────────────────────────────────────────────────
# (image_dir, label_dir, prefix)
# prefix is used to namespace filenames

TRAINING_SOURCES = [
    # Main raw dataset — labels in labels_poly/
    {
        "images": Path("data/raw/Positive"),
        "labels": Path("data/labels_poly/Positive"),
        "prefix": "pos",
    },
    {
        "images": Path("data/raw/Positive1"),
        "labels": Path("data/labels_poly/Positive1"),
        "prefix": "pos1",
    },
    {
        "images": Path("data/raw/BadTests"),
        "labels": Path("data/labels_poly/BadTests"),
        "prefix": "bad",
    },
    {
        "images": Path("data/raw/outside"),
        "labels": Path("data/labels_poly/outside"),
        "prefix": "out",
    },
    {
        "images": Path("data/raw/Negative"),
        "labels": Path("data/labels_poly/Negative"),
        "prefix": "neg",
    },
    {
        "images": Path("data/raw/Negative1"),
        "labels": Path("data/labels_poly/Negative1"),
        "prefix": "neg1",
    },
    {
        "images": Path("data/raw/Negative2"),
        "labels": Path("data/labels_poly/Negative2"),
        "prefix": "neg2",
    },
    # Screenshare with POI
    {
        "images": Path("data/raw/screenshare images/images"),
        "labels": Path("data/raw/screenshare images/labels"),
        "prefix": "ss",
    },
    # Screenshare without POI
    {
        "images": Path("data/raw/screenshare images no poi/images"),
        "labels": Path("data/raw/screenshare images no poi/labels"),
        "prefix": "ssneg",
    },
    # Auto-labeled
    {
        "images": Path("data/autolabel/images"),
        "labels": Path("data/autolabel/labels"),
        "prefix": "auto",
    },
]

TEST_DATA_SOURCES = [
    {
        "src": Path("data/test data/POI exists"),
        "dest": Path("data/test_data/with_poi"),
        "prefix": "tpos",
    },
    {
        "src": Path("data/test data/POI does not exist"),
        "dest": Path("data/test_data/without_poi"),
        "prefix": "tneg",
    },
]

OLD_DIRS_TO_ARCHIVE = [
    Path("data/raw"),
    Path("data/labels"),
    Path("data/labels_poly"),
    Path("data/labels_poly_smooth"),
    Path("data/autolabel"),
    Path("data/test data"),
    Path("outputs/seg_dataset"),
    Path("outputs/poi_seg_data.yaml"),
]


def collect_training_pairs() -> list[tuple[Path, Path, str]]:
    """Collect (image, label, new_name) tuples from all training sources."""
    pairs: list[tuple[Path, Path, str]] = []

    for src in TRAINING_SOURCES:
        img_dir = src["images"]
        lbl_dir = src["labels"]
        prefix = src["prefix"]

        if not img_dir.is_dir():
            print(f"  SKIP {img_dir} (not found)")
            continue

        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in SUPPORTED:
                continue

            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            new_stem = f"{prefix}_{img_path.stem}"
            pairs.append((img_path, label_path, new_stem))
            count += 1

        print(f"  {prefix:6s} ({img_dir}): {count} pairs")

    return pairs


def split_and_copy(pairs: list, dry_run: bool) -> None:
    """Shuffle, split, and copy files into data/images/{train,val,test}/."""
    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    print(f"\n  Split: {n_train} train / {n_val} val / {n - n_train - n_val} test")

    for split_name, split_pairs in splits.items():
        img_dir = Path(f"data/images/{split_name}")
        lbl_dir = Path(f"data/labels/{split_name}")

        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path, new_stem in split_pairs:
            # Normalize to .jpg
            dst_img = img_dir / f"{new_stem}.jpg"
            dst_lbl = lbl_dir / f"{new_stem}.txt"

            if dry_run:
                continue

            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

        print(f"  {split_name}: {len(split_pairs)} images -> {img_dir}")


def copy_test_data(dry_run: bool) -> None:
    """Copy test data (no labels) into data/test_data/."""
    for src in TEST_DATA_SOURCES:
        src_dir = src["src"]
        dest_dir = src["dest"]
        prefix = src["prefix"]

        if not src_dir.is_dir():
            print(f"  SKIP {src_dir} (not found)")
            continue

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in sorted(src_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in SUPPORTED:
                continue

            dst = dest_dir / f"{prefix}_{img_path.stem}.jpg"
            if not dry_run:
                shutil.copy2(img_path, dst)
            count += 1

        print(f"  {prefix}: {count} images -> {dest_dir}")


def archive_old_dirs(dry_run: bool) -> None:
    """Move old directories into data/_archive/."""
    archive_dir = Path("data/_archive")

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for old_dir in OLD_DIRS_TO_ARCHIVE:
        if not old_dir.exists():
            continue
        dest = archive_dir / old_dir.name
        if dest.exists():
            print(f"  SKIP {old_dir} -> {dest} (already archived)")
            continue
        if dry_run:
            print(f"  MOVE {old_dir} -> {dest}")
        else:
            shutil.move(str(old_dir), str(dest))
            print(f"  MOVED {old_dir} -> {dest}")


def reorganize_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Reorganize dataset into clean structure")
    parser.add_argument("--run", action="store_true",
                        help="Actually move files (default is dry run)")
    args = parser.parse_args()

    mode = "LIVE" if args.run else "DRY RUN"
    print(f"=== Dataset Reorganization ({mode}) ===\n")

    # Step 0: Archive old dirs FIRST so they don't collide with new structure
    if args.run:
        print("Step 0: Archiving old directories...")
        archive_old_dirs(dry_run=False)

    print("Step 1: Collecting training pairs...")
    pairs = collect_training_pairs()
    print(f"\n  Total training pairs: {len(pairs)}")

    print("\nStep 2: Splitting and copying...")
    split_and_copy(pairs, dry_run=not args.run)

    print("\nStep 3: Copying test data...")
    copy_test_data(dry_run=not args.run)

    print(f"\nDone ({mode}).")
    if not args.run:
        print("Run with --run to actually reorganize.")

