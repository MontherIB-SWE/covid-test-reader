"""Windows HWND helpers for screen-capture exclusion."""
from __future__ import annotations

import sys

from studio.config import _set_capture_exclusion


def apply_capture_exclusion_for_hwnd(hwnd: int, exclude: bool) -> bool:
    if sys.platform != "win32" or hwnd == 0:
        return False
    return _set_capture_exclusion(hwnd, exclude)
