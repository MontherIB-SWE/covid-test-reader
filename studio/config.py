"""Shared constants, theme, Win32 capture exclusion."""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Paths ──────────────────────────────────────────────────────────────────
# All project paths are defined here.  Every other module imports from this
# file — no hardcoded paths anywhere else.

# Data — single tree under data/
DATA_DIR = Path("data")
DATA_LABELED_DIR = DATA_DIR / "labeled"           # hand-labeled POI images
DATA_NEGATIVES_DIR = DATA_DIR / "negatives"        # backgrounds without POI
DATA_GENERATED_DIR = DATA_DIR / "generated"        # synthetic composites
DATA_ASSETS_DIR = DATA_DIR / "assets"              # extracted POI crops (PNG)
DATA_AUTOLABEL_DIR = DATA_DIR / "autolabel"        # auto-captured frames
DATA_UNLABELED_DIR = DATA_DIR / "unlabeled"       # POI images not yet labeled

# Outputs — build artifacts, configs, training bundles
OUTPUTS_DIR = Path("outputs")
TRAIN_SOURCES_JSON = OUTPUTS_DIR / "train_sources.json"
TRAIN_BUNDLE_DIR = OUTPUTS_DIR / "train_bundle"
TRAIN_DATA_YAML = OUTPUTS_DIR / "train_data.yaml"
DATASET_PATHS_JSON = OUTPUTS_DIR / "dataset_paths.json"

# Training runs — YOLO writes weights here
RUNS_DIR = Path("runs")

# Default model — searched first; resolve_model_path() finds the latest
DEFAULT_MODEL_PATH = RUNS_DIR / "poi_train_exp" / "weights" / "best.pt"

# ── Training defaults ──────────────────────────────────────────────────────
DEFAULT_LR = 0.002
DEFAULT_BATCH = 4
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 960
DEFAULT_PATIENCE = 30
LIVE_TARGET_FPS = 30
LIVE_TARGET_MS = int(1000 / LIVE_TARGET_FPS)
CAM_DISCONNECT_THRESHOLD = 10
AUTOLABEL_MIN_INTERVAL = 1.0

# ── Win32 screen capture ───────────────────────────────────────────────────
_WDA_NONE = 0x00000000
_WDA_EXCLUDEFROMCAPTURE = 0x00000011

# ── POI colors (BGR) ──────────────────────────────────────────────────────
POI_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
    (200, 200, 0),
    (128, 0, 128),
]
# Same list as POI_COLORS; name documents BGR semantics for callers.
POI_COLORS_BGR = POI_COLORS


def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


POI_COLORS_HEX = [_bgr_to_hex(c) for c in POI_COLORS_BGR]
DRAG_RADIUS = 8

# ── Theme ──────────────────────────────────────────────────────────────────
BG = "#0f0f0f"
BG_PANEL = "#1a1a1a"
BG_CARD = "#222222"
BG_INPUT = "#2a2a2a"
BG_HOVER = "#333333"
BG_ACTIVE = "#3a3a3a"
FG = "#e0e0e0"
FG_DIM = "#888888"
FG_SEL = "#ffffff"
ACCENT = "#00a86b"
ACCENT_HOVER = "#00c07e"
ORANGE = "#e0952a"
ORANGE_HOVER = "#f0a83e"
RED = "#e04040"
RED_HOVER = "#f05050"
YELLOW = "#e0c040"
CYAN = "#40c0e0"
BORDER = "#333333"
SEP = "#2a2a2a"
FONT = "Segoe UI"
MONO = "Consolas"

# ── Win32 bindings ─────────────────────────────────────────────────────────
_SetWindowDisplayAffinity = None
_GetAncestor = None

if sys.platform == "win32":
    try:
        _user32 = ctypes.WinDLL("user32", use_last_error=True)
        _SetWindowDisplayAffinity = _user32.SetWindowDisplayAffinity
        _SetWindowDisplayAffinity.argtypes = (ctypes.wintypes.HWND, ctypes.wintypes.DWORD)
        _SetWindowDisplayAffinity.restype = ctypes.wintypes.BOOL
        _GetAncestor = _user32.GetAncestor
        _GetAncestor.argtypes = (ctypes.wintypes.HWND, ctypes.wintypes.UINT)
        _GetAncestor.restype = ctypes.wintypes.HWND
    except Exception:
        pass

_GA_ROOT = 2


def _get_toplevel_hwnd(widget_hwnd: int) -> int:
    if _GetAncestor is not None:
        return _GetAncestor(widget_hwnd, _GA_ROOT) or widget_hwnd
    return widget_hwnd


def _set_capture_exclusion(widget_hwnd: int, exclude: bool) -> bool:
    if _SetWindowDisplayAffinity is None:
        return False
    hwnd = _get_toplevel_hwnd(widget_hwnd)
    affinity = _WDA_EXCLUDEFROMCAPTURE if exclude else _WDA_NONE
    try:
        return bool(_SetWindowDisplayAffinity(hwnd, affinity))
    except Exception:
        return False


def _select_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def resolve_model_path() -> Path | None:
    """Find the most recent trained model across all run directories.

    Searches RUNS_DIR for best.pt files, returning the one with the newest
    modification time.  Falls back to DEFAULT_MODEL_PATH if it exists.
    Returns None when no model is found.
    """
    candidates: list[Path] = []

    if RUNS_DIR.is_dir():
        try:
            candidates.extend(RUNS_DIR.rglob("best.pt"))
        except OSError:
            pass

    if DEFAULT_MODEL_PATH.is_file():
        candidates.append(DEFAULT_MODEL_PATH)

    if not candidates:
        return None

    # Deduplicate
    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return max(unique, key=lambda p: p.stat().st_mtime)
