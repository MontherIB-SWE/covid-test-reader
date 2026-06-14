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
DATA_RESULT_CLS_DIR = DATA_DIR / "result_cls"     # result classification data
RESULT_CLS_MANIFEST = DATA_RESULT_CLS_DIR / "manifest.csv"
RESULT_CLS_CROPS_DIR = DATA_RESULT_CLS_DIR / "crops"
RESULT_CLS_AUGMENTED_DIR = DATA_RESULT_CLS_DIR / "augmented"
RESULT_CLS_AUGMENT_INDEX = DATA_RESULT_CLS_DIR / "augmented_index.json"
RESULT_CLS_AUG_PER_STEM = 12
RESULT_CLS_LIVE_LOG = DATA_RESULT_CLS_DIR / "live_corrections.jsonl"
RESULT_CLS_LIVE_SESSION_JSON = DATA_RESULT_CLS_DIR / "last_live_session.json"

# Training bundle scope (result classifier)
SCOPE_CURRENT_LIVE = "current_live_session"
SCOPE_ALL_LIVE = "all_live"
SCOPE_FULL = "full"
SCOPE_LABELED_ONLY = "labeled_only"

# Outputs — build artifacts, configs, training bundles
OUTPUTS_DIR = Path("outputs")
TRAIN_SOURCES_JSON = OUTPUTS_DIR / "train_sources.json"
TRAIN_BUNDLE_DIR = OUTPUTS_DIR / "train_bundle"
TRAIN_DATA_YAML = OUTPUTS_DIR / "train_data.yaml"
DATASET_PATHS_JSON = OUTPUTS_DIR / "dataset_paths.json"
RESULT_CLS_BUNDLE_DIR = OUTPUTS_DIR / "result_cls_bundle"

# Training runs — YOLO writes weights here
RUNS_DIR = Path("runs")

# Default model — searched first; resolve_model_path() finds the latest
DEFAULT_MODEL_PATH = RUNS_DIR / "poi_train_exp" / "weights" / "best.pt"

# Result classifier (ResNet-50)
RESULT_CLS_RUNS_DIR = RUNS_DIR / "result_cls"
RESULT_CLS_ARCH = "resnet50"

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
BG = "#090a0f"
BG_PANEL = "#11131e"
BG_CARD = "#171a27"
BG_INPUT = "#1f2336"
BG_HOVER = "#282c44"
BG_ACTIVE = "#303652"
FG = "#f8fafc"
FG_DIM = "#94a3b8"
FG_SEL = "#ffffff"
ACCENT = "#6366f1"
ACCENT_HOVER = "#4f46e5"
ORANGE = "#f59e0b"
ORANGE_HOVER = "#d97706"
RED = "#ef4444"
RED_HOVER = "#dc2626"
YELLOW = "#eab308"
CYAN = "#06b6d4"
BORDER = "#25293d"
SEP = "#1e2235"
FONT = "Inter, Segoe UI, system-ui, sans-serif"
MONO = "Consolas, Fira Code, monospace"

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


def resolve_result_cls_model_path() -> Path | None:
    """Find the most recent ResNet-34 result classifier checkpoint."""
    if not RESULT_CLS_RUNS_DIR.is_dir():
        return None
    candidates: list[Path] = []
    try:
        candidates.extend(RESULT_CLS_RUNS_DIR.rglob("best.pt"))
    except OSError:
        pass
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
