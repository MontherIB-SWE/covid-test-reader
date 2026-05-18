"""NumPy / OpenCV image helpers for Qt."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def numpy_bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def numpy_bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_bgr_to_qimage(bgr))


def fit_pixmap(pm: QPixmap, max_w: int, max_h: int) -> QPixmap:
    if pm.isNull():
        return pm
    return pm.scaled(
        max_w,
        max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
