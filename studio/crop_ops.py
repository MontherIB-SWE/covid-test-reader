"""Shared crop utilities: quad ordering, perspective warp, letterbox."""
from __future__ import annotations

import cv2
import numpy as np

RESULT_CLS_IMGSZ = 448


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def warp_quad_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp a quadrilateral region to a rectangular image."""
    tl, tr, br, bl = quad
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    out_w = max(1, int(round(max(width_top, width_bottom))))
    out_h = max(1, int(round(max(height_right, height_left))))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, matrix, (out_w, out_h))


def letterbox(bgr: np.ndarray, size: int = RESULT_CLS_IMGSZ) -> np.ndarray:
    """Resize with aspect ratio preserved, pad to square with gray (114)."""
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
