from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Callable
import ctypes
import ctypes.wintypes
import os
import queue
import re
import shutil
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_PATH = Path("runs/obb_v1/poi_obb_v1/weights/best.pt")

# Training tab (paths relative to cwd when launching the app)
DATA_MASTER = Path("data_master")
OUTPUTS_DIR = Path("outputs")
RUNS_DIR = Path("runs")
DEFAULT_LR = 0.002
DEFAULT_BATCH = 4
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 960
DEFAULT_PATIENCE = 30
LIVE_TARGET_FPS = 30
LIVE_TARGET_MS = int(1000 / LIVE_TARGET_FPS)
CAM_DISCONNECT_THRESHOLD = 10
AUTOLABEL_MIN_INTERVAL = 1.0  # seconds between auto-saved frames

# Windows Display Affinity constants
_WDA_NONE = 0x00000000
_WDA_EXCLUDEFROMCAPTURE = 0x00000011

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

# ── Theme ────────────────────────────────────────────────────────────────
BG        = "#0f0f0f"
BG_PANEL  = "#1a1a1a"
BG_CARD   = "#222222"
BG_INPUT  = "#2a2a2a"
BG_HOVER  = "#333333"
BG_ACTIVE = "#3a3a3a"
FG        = "#e0e0e0"
FG_DIM    = "#888888"
ACCENT    = "#00a86b"
ACCENT_HOVER = "#00c07e"
ORANGE    = "#e0952a"
ORANGE_HOVER = "#f0a83e"
RED       = "#e04040"
RED_HOVER    = "#f05050"
BORDER    = "#333333"
SEP       = "#2a2a2a"
FONT      = "Segoe UI"
MONO      = "Consolas"

# ── Windows screen-capture exclusion ─────────────────────────────────────

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
    """Walk up to the real top-level Win32 window.

    ``tk.winfo_id()`` returns a child-widget HWND on Windows, but
    ``SetWindowDisplayAffinity`` only works on the top-level window.
    ``GetAncestor(hwnd, GA_ROOT)`` gives us the correct handle.
    """
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


# ── Styled widgets ───────────────────────────────────────────────────────

class FlatButton(tk.Canvas):
    def __init__(
        self, parent, *, text="", command=None, bg=ACCENT, bg_hover=ACCENT_HOVER,
        fg="#ffffff", width=90, height=32, font_size=9, bold=False, radius=6,
    ) -> None:
        self._bg = bg
        self._bg_hover = bg_hover
        self._fg = fg
        self._text = text
        self._command = command
        self._radius = radius
        weight = "bold" if bold else "normal"
        self._font = (FONT, font_size, weight)
        parent_bg = BG_PANEL
        if isinstance(parent, (tk.Frame, tk.Canvas)):
            try:
                parent_bg = parent.cget("bg")
            except Exception:
                pass
        super().__init__(parent, width=width, height=height,
                         bg=parent_bg, highlightthickness=0, bd=0)
        self._draw(self._bg)
        self.bind("<Enter>", lambda _: self._draw(self._bg_hover))
        self.bind("<Leave>", lambda _: self._draw(self._bg))
        self.bind("<ButtonPress-1>", lambda _: self._click())
        self.bind("<ButtonRelease-1>", lambda _: self._draw(self._bg_hover))

    def _draw(self, color: str) -> None:
        self.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        r = self._radius
        pts = [
            x1 + r, y1, x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r, x1, y1,
        ] if False else self._rrect_pts(0, 0, w, h, r)
        self.create_polygon(pts, smooth=True, fill=color, outline="")
        self.create_text(w // 2, h // 2, text=self._text,
                         fill=self._fg, font=self._font)

    @staticmethod
    def _rrect_pts(x1, y1, x2, y2, r):
        return [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1,
        ]

    def _click(self) -> None:
        if self._command:
            self._command()

    def configure_text(self, text: str) -> None:
        self._text = text
        self._draw(self._bg)

    def configure_colors(self, *, bg: str | None = None, bg_hover: str | None = None,
                         fg: str | None = None) -> None:
        if bg is not None:
            self._bg = bg
        if bg_hover is not None:
            self._bg_hover = bg_hover
        if fg is not None:
            self._fg = fg
        self._draw(self._bg)


class FlatLabel(tk.Label):
    def __init__(self, parent, **kw) -> None:
        kw.setdefault("bg", BG_PANEL)
        kw.setdefault("fg", FG)
        kw.setdefault("font", (FONT, 9))
        super().__init__(parent, **kw)


class FlatOptionMenu(tk.Frame):
    def __init__(self, parent, variable: tk.StringVar, *values, width=10) -> None:
        super().__init__(parent, bg=BG_INPUT, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT)
        self._var = variable
        self._values = list(values)
        self._menu_open = False
        self._list_win: tk.Toplevel | None = None

        self._label = tk.Label(self, textvariable=variable, bg=BG_INPUT, fg=FG,
                               font=(FONT, 9), anchor="w", padx=8)
        self._label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=5)

        arrow = tk.Label(self, text="\u25BE", bg=BG_INPUT, fg=FG_DIM, font=(FONT, 10))
        arrow.pack(side=tk.RIGHT, padx=(0, 6))

        for w in (self, self._label, arrow):
            w.bind("<ButtonPress-1>", self._toggle)

    def _toggle(self, _event=None) -> None:
        if self._menu_open and self._list_win:
            self._close_menu()
            return
        self._open_menu()

    def _open_menu(self) -> None:
        if not self._values:
            return
        self._menu_open = True
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        win = tk.Toplevel(self)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=BG_CARD)
        win.geometry(f"+{x}+{y}")
        for val in self._values:
            lbl = tk.Label(win, text=val, bg=BG_CARD, fg=FG,
                           font=(FONT, 9), anchor="w", padx=14, pady=5, cursor="hand2")
            lbl.pack(fill=tk.X)
            lbl.bind("<Enter>", lambda e, l=lbl: l.configure(bg=BG_HOVER))
            lbl.bind("<Leave>", lambda e, l=lbl: l.configure(bg=BG_CARD))
            lbl.bind("<ButtonPress-1>", lambda e, v=val: self._pick(v))
        win.bind("<FocusOut>", lambda _: self._close_menu())
        self._list_win = win
        win.focus_set()

    def _pick(self, value: str) -> None:
        self._var.set(value)
        self._close_menu()

    def _close_menu(self) -> None:
        self._menu_open = False
        if self._list_win:
            self._list_win.destroy()
            self._list_win = None

    def update_values(self, values: list[str]) -> None:
        self._values = values


# ── Main application ─────────────────────────────────────────────────────

class PoiDesktopViewer:
    def __init__(self, root: tk.Misc, *, embedded: bool = False, bind_shortcuts: bool = True) -> None:
        self.root = root
        self.embedded = embedded
        self.window = root if isinstance(root, tk.Tk) else root.winfo_toplevel()
        if not self.embedded:
            self.window.title("POI Desktop Viewer")
            self.window.geometry("1440x860")
            self.window.minsize(1100, 650)
            self.window.configure(bg=BG)

        self.device = _select_device()
        self.model_path = DEFAULT_MODEL_PATH
        self.model: YOLO | None = None
        self.current_image_path: Path | None = None

        self.image_paths: list[Path] = []
        self.index = 0

        self.left_photo: ImageTk.PhotoImage | None = None
        self._crop_photos: list[ImageTk.PhotoImage] = []
        self._crops_canvas: tk.Canvas | None = None
        self._crops_inner: tk.Frame | None = None
        self.live_running = False
        self.live_job: str | None = None
        self.camera_capture: cv2.VideoCapture | None = None
        self.active_live_source: str | None = None
        self.live_source_var = tk.StringVar(value="Camera")
        self.camera_index_var = tk.StringVar(value="0")
        self.available_cameras: list[str] = []
        self.camera_menu: FlatOptionMenu | None = None

        # Auto-label state
        self.autolabel_active = False
        self.autolabel_dir = Path("data/autolabel_output")
        self._autolabel_count = 0
        self._autolabel_last_save = 0.0
        self._autolabel_btn: FlatButton | None = None
        self._autolabel_counter_lbl: tk.Label | None = None

        # Background inference
        self._inference_pending = False
        self._camera_lock = threading.Lock()
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._camera_reader_thread: threading.Thread | None = None
        self._camera_reader_stop = threading.Event()
        self._camera_dead = False
        # Tk 3.13+: do not call root.after() from worker threads; use queues + main-thread poll.
        self._camera_results_queue: queue.Queue[list[str]] = queue.Queue()
        # ("ok", overlay, crops, status, raw_frame, result) | ("err", message)
        self._live_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._screen_excluded = False
        self._last_infer_frame: np.ndarray | None = None
        self._last_infer_result = None
        self._capture_for_relabel_callback = None

        self._build_ui()
        if bind_shortcuts:
            self._bind_keys()
        self._bind_source_trace()
        self._load_model()

        # Ensure window is mapped before applying exclusion
        self.root.update_idletasks()
        self.root.after(100, self._apply_initial_exclusion)

        if not self.embedded:
            self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_camera_list_background()

    def _apply_initial_exclusion(self) -> None:
        """Apply screen exclusion if the source starts as Screen (rare) or refresh state."""
        pass  # Exclusion is applied when live starts

    # ─────────────────────────────────────────────────────────────── UI ──

    def _build_ui(self) -> None:
        # ── Header bar ──
        header = tk.Frame(self.root, bg=BG_PANEL, height=54)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        row1 = tk.Frame(header, bg=BG_PANEL, padx=14, pady=6)
        row1.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(row1, text="POI", bg=BG_PANEL, fg=ACCENT,
                 font=(FONT, 15, "bold")).pack(side=tk.LEFT)
        tk.Label(row1, text="Viewer", bg=BG_PANEL, fg=FG,
                 font=(FONT, 15, "bold")).pack(side=tk.LEFT, padx=(0, 16))

        self._sep(row1)

        # File buttons
        FlatButton(row1, text="Open Image", command=self._open_image,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=88, height=30, font_size=9
                   ).pack(side=tk.LEFT, padx=(8, 3))
        FlatButton(row1, text="Open Folder", command=self._open_folder,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=88, height=30, font_size=9
                   ).pack(side=tk.LEFT, padx=3)

        self._sep(row1)

        # Navigation
        FlatButton(row1, text="\u25C0", command=self._prev_image,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=30, height=30, font_size=11
                   ).pack(side=tk.LEFT, padx=(8, 2))
        FlatButton(row1, text="\u25B6", command=self._next_image,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=30, height=30, font_size=11
                   ).pack(side=tk.LEFT, padx=(2, 0))

        self._sep(row1)

        # Model
        FlatButton(row1, text="Model", command=self._choose_model,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=62, height=30, font_size=9
                   ).pack(side=tk.LEFT, padx=(8, 0))

        FlatButton(row1, text="Load Last", command=self._load_last_trained,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=72, height=30, font_size=9
                   ).pack(side=tk.LEFT, padx=(8, 0))

        self._sep(row1)

        # Live source controls
        FlatLabel(row1, text="Source").pack(side=tk.LEFT, padx=(8, 3))
        FlatOptionMenu(row1, self.live_source_var, "Camera", "Screen", width=8
                       ).pack(side=tk.LEFT, padx=(0, 6))

        FlatLabel(row1, text="Cam").pack(side=tk.LEFT, padx=(0, 3))
        self.camera_menu = FlatOptionMenu(row1, self.camera_index_var, "0", width=4)
        self.camera_menu.pack(side=tk.LEFT, padx=(0, 3))
        FlatButton(row1, text="\u21BB", command=self._refresh_camera_list_background,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=30, height=30, font_size=12
                   ).pack(side=tk.LEFT, padx=(0, 6))

        # Start / Stop
        FlatButton(row1, text="\u25B6 Start", command=self._start_live,
                   bg=ACCENT, bg_hover=ACCENT_HOVER, width=70, height=30, bold=True, font_size=9
                   ).pack(side=tk.LEFT, padx=(0, 3))
        FlatButton(row1, text="\u25A0 Stop", command=self._stop_live,
                   bg=RED, bg_hover=RED_HOVER, width=62, height=30, bold=True, font_size=9
                   ).pack(side=tk.LEFT, padx=(0, 0))

        self._sep(row1)

        # Auto-Label toggle
        self._autolabel_btn = FlatButton(
            row1, text="Auto-Label: OFF", command=self._toggle_autolabel,
            bg=BG_CARD, bg_hover=BG_HOVER, fg=FG_DIM,
            width=116, height=30, font_size=9, bold=True,
        )
        self._autolabel_btn.pack(side=tk.LEFT, padx=(8, 3))

        FlatButton(row1, text="Output Dir", command=self._choose_autolabel_dir,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=72, height=30, font_size=9
                   ).pack(side=tk.LEFT, padx=(0, 6))
        FlatButton(row1, text="Capture \u2192 Relabel", command=self._capture_for_relabel,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=110, height=30, font_size=9
                   ).pack(side=tk.LEFT, padx=(0, 6))

        # Device badge + counter (right side)
        self._autolabel_counter_lbl = tk.Label(
            row1, text="", bg=BG_PANEL, fg=ORANGE, font=(MONO, 9, "bold"))
        self._autolabel_counter_lbl.pack(side=tk.RIGHT, padx=(0, 4))

        self._device_lbl = tk.Label(
            row1, text=f" {self.device.upper()} ", bg=BG_CARD,
            fg=ACCENT, font=(MONO, 8, "bold"), padx=6, pady=2)
        self._device_lbl.pack(side=tk.RIGHT, padx=(4, 0))

        # ── Status bar ──
        status_bar = tk.Frame(self.root, bg=BG_PANEL, height=40)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        status_inner = tk.Frame(status_bar, bg=BG_PANEL, padx=14)
        status_inner.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(status_inner, textvariable=self.status_var, bg=BG_PANEL,
                 fg=FG_DIM, font=(MONO, 10), anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True, pady=6)

        self._autolabel_path_lbl = tk.Label(
            status_inner, text=f"\U0001F4C2 {self.autolabel_dir}", bg=BG_PANEL,
            fg=FG_DIM, font=(MONO, 8), anchor="e")
        self._autolabel_path_lbl.pack(side=tk.RIGHT, padx=(8, 0), pady=4)

        # ── Viewport area ──
        viewport = tk.Frame(self.root, bg=BG)
        viewport.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        viewport.grid_columnconfigure(0, weight=1)
        viewport.grid_columnconfigure(1, weight=1)
        viewport.grid_rowconfigure(1, weight=1)

        for col, title in enumerate(["Image + POI Overlay", "POI Crops"]):
            hdr = tk.Frame(viewport, bg=BG)
            hdr.grid(row=0, column=col, sticky="ew", pady=(0, 4))
            tk.Frame(hdr, bg=ACCENT, width=3, height=14).pack(side=tk.LEFT, padx=(0, 8))
            tk.Label(hdr, text=title, bg=BG, fg=FG, font=(FONT, 10, "bold")).pack(side=tk.LEFT)

        # Left panel — single label for overlay
        left_card = tk.Frame(viewport, bg=BORDER, bd=0)
        left_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.left_label = tk.Label(left_card, bg=BG_CARD, bd=0)
        self.left_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Right panel — scrollable crop grid
        right_card = tk.Frame(viewport, bg=BORDER, bd=0)
        right_card.grid(row=1, column=1, sticky="nsew", padx=(0, 0))

        self._crops_canvas = tk.Canvas(right_card, bg=BG_CARD, bd=0,
                                       highlightthickness=0, width=400)
        vscroll = tk.Scrollbar(right_card, orient=tk.VERTICAL,
                               command=self._crops_canvas.yview)
        self._crops_canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0), pady=1)
        self._crops_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(1, 0), pady=1)

        self._crops_inner = tk.Frame(self._crops_canvas, bg=BG_CARD)
        self._crops_canvas.create_window((0, 0), window=self._crops_inner, anchor="nw",
                                         tags="inner")
        self._crops_inner.bind("<Configure>",
                               lambda _: self._crops_canvas.configure(
                                   scrollregion=self._crops_canvas.bbox("all")))
        self._crops_canvas.bind("<Configure>", self._on_crops_canvas_resize)

        # Mousewheel scrolling on crops panel
        def _on_mousewheel(event):
            self._crops_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._crops_canvas.bind("<MouseWheel>", _on_mousewheel)
        self._crops_inner.bind("<MouseWheel>", _on_mousewheel)

    @staticmethod
    def _sep(parent) -> None:
        tk.Frame(parent, bg=SEP, width=1, height=22).pack(side=tk.LEFT, padx=6, pady=4)

    def _bind_keys(self) -> None:
        self.window.bind("<Left>", lambda _: self._prev_image())
        self.window.bind("<Right>", lambda _: self._next_image())
        self.window.bind("<space>", lambda _: self._toggle_autolabel())

    def _bind_source_trace(self) -> None:
        self.live_source_var.trace_add("write", self._on_live_source_changed)

    def _on_crops_canvas_resize(self, event) -> None:
        self._crops_canvas.itemconfigure("inner", width=event.width)

    def _on_live_source_changed(self, *_args) -> None:
        if not self.live_running:
            return
        new_source = self.live_source_var.get()
        if new_source == self.active_live_source:
            return
        self._stop_camera_reader()
        self._close_camera()
        self._remove_screen_exclusion()
        if new_source == "Camera":
            if not self._open_camera_capture():
                self.status_var.set("Camera open failed after source switch")
            else:
                self._start_camera_reader()
        elif new_source == "Screen":
            self._apply_screen_exclusion()
        self.active_live_source = new_source

    # ─────────────────────────────── Screen capture exclusion ──

    def _apply_screen_exclusion(self) -> None:
        self.root.update_idletasks()
        hwnd = self.root.winfo_id()
        ok = _set_capture_exclusion(hwnd, True)
        self._screen_excluded = ok
        if ok:
            self.status_var.set("Screen exclusion active \u2014 app hidden from capture")
        else:
            self.status_var.set("Screen exclusion not available (requires Windows 10 2004+)")

    def _remove_screen_exclusion(self) -> None:
        if self._screen_excluded:
            _set_capture_exclusion(self.root.winfo_id(), False)
            self._screen_excluded = False

    # ──────────────────────────────────────────────────────────── Model ──

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.status_var.set(f"Model not found: {self.model_path}  |  device={self.device}")
            return
        try:
            self.model = YOLO(str(self.model_path))
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.predict(source=dummy, device=self.device, conf=0.50, imgsz=960, verbose=False)
            self.status_var.set(f"Loaded {self.model_path.name}  |  device={self.device}")
        except Exception as exc:
            self.model = None
            self.status_var.set(f"Failed to load model: {exc}")

    def _choose_model(self) -> None:
        self._stop_live()
        model_file = filedialog.askopenfilename(
            title="Choose YOLO model (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not model_file:
            return
        self.model_path = Path(model_file)
        self._load_model()
        if self.current_image_path and self.model is not None:
            self._run_prediction(self.current_image_path)

    # ───────────────────────────────────────────────────────── File I/O ──

    def _open_image(self) -> None:
        self._stop_live()
        image_file = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not image_file:
            return
        image_path = Path(image_file)
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            messagebox.showerror("Unsupported file", f"Unsupported image extension: {image_path.suffix}")
            return
        self.image_paths = []
        self.current_image_path = image_path
        self._run_prediction(image_path)

    def _open_folder(self) -> None:
        self._stop_live()
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        base = Path(folder)
        images = sorted(
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            messagebox.showwarning("No images", "No supported images found in folder.")
            return
        self.image_paths = images
        self.index = 0
        self._show_current()

    def _prev_image(self) -> None:
        if not self.image_paths:
            return
        self.index = max(0, self.index - 1)
        self._show_current()

    def _next_image(self) -> None:
        if not self.image_paths:
            return
        self.index = min(len(self.image_paths) - 1, self.index + 1)
        self._show_current()

    def _show_current(self) -> None:
        if not self.image_paths:
            return
        image_path = self.image_paths[self.index]
        self.current_image_path = image_path
        self._run_prediction(image_path)

    # ────────────────────────────────────────────────────── Prediction ──

    def _run_prediction(self, image_path: Path) -> None:
        if self.model is None:
            messagebox.showerror("Model not loaded", "Load a valid .pt model first.")
            return
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            messagebox.showerror("Image error", f"Could not read image: {image_path}")
            return
        tag = f"[{self.index + 1}/{len(self.image_paths)}]" if self.image_paths else ""
        self.status_var.set(f"{tag} Predicting: {image_path.name} ...")
        self.root.update_idletasks()
        try:
            result = self.model.predict(source=bgr, device=self.device, conf=0.50, imgsz=960, verbose=False)[0]
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return
        overlay, crops, status = self._build_outputs(bgr, result)
        self._last_infer_frame = bgr.copy()
        self._last_infer_result = result
        self.status_var.set(status)
        self._set_preview_images(overlay, crops)

    # ────────────────────────────────────────────────────────── Camera ──

    def _open_camera_capture(self) -> bool:
        try:
            cam_index = int(self.camera_index_var.get().strip())
        except ValueError:
            messagebox.showerror("Camera index", "Camera index must be an integer.")
            return False
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cam_index, backend)
        if not cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera index {cam_index}.")
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        with self._camera_lock:
            self.camera_capture = cap
        return True

    def _close_camera(self) -> None:
        with self._camera_lock:
            if self.camera_capture is not None:
                self.camera_capture.release()
                self.camera_capture = None

    def _start_camera_reader(self) -> None:
        self._camera_reader_stop.clear()
        self._camera_dead = False
        with self._latest_frame_lock:
            self._latest_frame = None
        self._camera_reader_thread = threading.Thread(
            target=self._camera_reader_loop, daemon=True,
        )
        self._camera_reader_thread.start()

    def _stop_camera_reader(self) -> None:
        self._camera_reader_stop.set()
        if self._camera_reader_thread is not None:
            self._camera_reader_thread.join(timeout=2.0)
            self._camera_reader_thread = None

    def _camera_reader_loop(self) -> None:
        fail_count = 0
        while not self._camera_reader_stop.is_set():
            with self._camera_lock:
                cap = self.camera_capture
                if cap is None:
                    break
                ok, frame = cap.read()
            if ok and frame is not None:
                with self._latest_frame_lock:
                    self._latest_frame = frame
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= CAM_DISCONNECT_THRESHOLD:
                    self._camera_dead = True
                    break
                time.sleep(0.005)

    def _detect_camera_indices(self, max_index: int = 3) -> list[str]:
        detected: list[str] = []
        for idx in range(max_index):
            cap = None
            try:
                # Redirect C-level stderr (fd 2) to suppress DirectShow assertions
                old_stderr = os.dup(2)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                finally:
                    os.dup2(old_stderr, 2)
                    os.close(old_stderr)
                    os.close(devnull)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        detected.append(str(idx))
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
        return detected

    def _refresh_camera_list_background(self) -> None:
        self.status_var.set("Scanning for cameras...")

        def _worker() -> None:
            cameras = self._detect_camera_indices()
            self._camera_results_queue.put(cameras)

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_camera_scan_results()

    def _poll_camera_scan_results(self) -> None:
        latest: list[str] | None = None
        try:
            while True:
                latest = self._camera_results_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            try:
                if self.window.winfo_exists():
                    self.window.after(50, self._poll_camera_scan_results)
            except tk.TclError:
                pass
            return
        self._apply_camera_list(latest)

    def _apply_camera_list(self, cameras: list[str]) -> None:
        if not cameras:
            cameras = ["0"]
            self.status_var.set("No camera detected. Using default index 0.")
        else:
            self.status_var.set(f"Found cameras: {', '.join(cameras)}")
        self.available_cameras = cameras
        if self.camera_index_var.get() not in cameras:
            self.camera_index_var.set(cameras[0])
        if self.camera_menu is not None:
            self.camera_menu.update_values(cameras)

    # ──────────────────────────────────────────────────────── Live loop ──

    def _start_live(self) -> None:
        if self.model is None:
            messagebox.showerror("Model not loaded", "Load a valid .pt model first.")
            return
        if self.live_running:
            return
        source = self.live_source_var.get()
        if source == "Camera":
            if not self._open_camera_capture():
                return
            self._start_camera_reader()
        elif source == "Screen":
            self._apply_screen_exclusion()
        self.active_live_source = source
        self._inference_pending = False
        self.live_running = True
        self._live_tick()

    def _stop_live(self) -> None:
        self.live_running = False
        if self.live_job is not None:
            self.root.after_cancel(self.live_job)
            self.live_job = None
        self._stop_camera_reader()
        self._close_camera()
        self._remove_screen_exclusion()
        self.active_live_source = None
        self._inference_pending = False

    def _read_live_frame(self) -> tuple[np.ndarray | None, str]:
        source = self.active_live_source or self.live_source_var.get()
        if source == "Screen":
            screen = ImageGrab.grab()
            rgb = np.array(screen)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr, "Live screen"
        if self._camera_dead:
            return None, "Camera disconnected"
        with self._latest_frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None, "Camera not ready"
        return frame.copy(), f"Live camera {self.camera_index_var.get().strip()}"

    def _drain_live_ui_queue(self) -> None:
        while True:
            try:
                item = self._live_ui_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "ok":
                _, overlay, crops, status, raw_frame, result = item
                self._deliver_live_result(overlay, crops, status, raw_frame, result)
            else:
                self._deliver_live_error(item[1])

    def _live_tick(self) -> None:
        if not self.live_running:
            return
        self._drain_live_ui_queue()
        t_start = time.perf_counter()
        if self._inference_pending:
            self.live_job = self.root.after(5, self._live_tick)
            return
        frame, label = self._read_live_frame()
        if frame is not None:
            frame_copy = frame.copy()
            self._inference_pending = True
            def _worker() -> None:
                try:
                    result = self.model.predict(
                        source=frame_copy, device=self.device, conf=0.50, imgsz=960, verbose=False,
                    )[0]
                    overlay, crops, status = self._build_outputs(frame_copy, result)
                    obb = getattr(result, "obb", None)
                    has_poi = obb is not None and len(obb) > 0
                    self._live_ui_queue.put(
                        (
                            "ok",
                            overlay,
                            crops,
                            f"{label} | {status}",
                            frame_copy if has_poi else None,
                            result if has_poi else None,
                        )
                    )
                except Exception as exc:
                    self._live_ui_queue.put(("err", str(exc)))
            threading.Thread(target=_worker, daemon=True).start()
        else:
            self.status_var.set(label)
            if self._camera_dead:
                self._stop_live()
                return
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        delay = max(1, LIVE_TARGET_MS - int(elapsed_ms))
        if self.live_running:
            self.live_job = self.root.after(delay, self._live_tick)

    def _deliver_live_result(self, overlay: np.ndarray, crops: np.ndarray,
                             status: str, raw_frame: np.ndarray | None,
                             result) -> None:
        self._inference_pending = False
        if not self.live_running:
            return
        self._set_preview_images(overlay, crops)
        self.status_var.set(status)
        self._last_infer_frame = raw_frame.copy() if raw_frame is not None else None
        self._last_infer_result = result
        if self.autolabel_active and raw_frame is not None and result is not None:
            self._autolabel_save(raw_frame, result)

    def _deliver_live_error(self, error: str) -> None:
        self._inference_pending = False
        if not self.live_running:
            return
        self.status_var.set(f"Prediction error: {error}")

    # ───────────────────────────────────────────── Auto-label ──

    def _toggle_autolabel(self) -> None:
        if self.autolabel_active:
            self.autolabel_active = False
            if self._autolabel_btn:
                self._autolabel_btn.configure_text("Auto-Label: OFF")
                self._autolabel_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER, fg=FG_DIM)
            self.status_var.set("Auto-label stopped")
        else:
            self.autolabel_dir.mkdir(parents=True, exist_ok=True)
            (self.autolabel_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.autolabel_dir / "labels").mkdir(parents=True, exist_ok=True)
            self.autolabel_active = True
            self._autolabel_last_save = 0.0
            if self._autolabel_btn:
                self._autolabel_btn.configure_text("Auto-Label: ON")
                self._autolabel_btn.configure_colors(bg=ORANGE, bg_hover=ORANGE_HOVER, fg="#ffffff")
            self._update_autolabel_counter()
            self.status_var.set(f"Auto-label active \u2192 {self.autolabel_dir}")

    def _choose_autolabel_dir(self) -> None:
        folder = filedialog.askdirectory(
            title="Choose auto-label output directory",
            initialdir=str(self.autolabel_dir.parent),
        )
        if not folder:
            return
        self.autolabel_dir = Path(folder)
        self._autolabel_count = 0
        self._update_autolabel_counter()
        self._autolabel_path_lbl.configure(text=f"\U0001F4C2 {self.autolabel_dir}")
        self.status_var.set(f"Auto-label output set to {self.autolabel_dir}")

    def _autolabel_save(self, bgr: np.ndarray, result) -> None:
        """Save raw frame + YOLO polygon labels if enough time has elapsed."""
        now = time.monotonic()
        if now - self._autolabel_last_save < AUTOLABEL_MIN_INTERVAL:
            return
        saved = self._save_prediction_as_label(bgr, result)
        if not saved:
            return
        self._autolabel_count += 1
        self._autolabel_last_save = now
        self._update_autolabel_counter()

    def _save_prediction_as_label(self, bgr: np.ndarray, result) -> tuple[Path, Path] | None:
        """Persist one image + label file (empty for negative frames)."""
        self.autolabel_dir.mkdir(parents=True, exist_ok=True)
        (self.autolabel_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.autolabel_dir / "labels").mkdir(parents=True, exist_ok=True)

        h, w = bgr.shape[:2]
        ts = int(time.time() * 1000)
        stem = str(ts)
        img_path = self.autolabel_dir / "images" / f"{stem}.jpg"
        lbl_path = self.autolabel_dir / "labels" / f"{stem}.txt"

        if not cv2.imwrite(str(img_path), bgr):
            return None

        lines: list[str] = []
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            # OBB outputs pixel-coordinate quads directly via xyxyxyxy
            for corners in obb.xyxyxyxy.cpu().numpy():
                quad = self._order_quad_points(corners.astype(np.float32))
                coords: list[str] = []
                for px, py in quad:
                    coords.append(f"{px / w:.6f}")
                    coords.append(f"{py / h:.6f}")
                lines.append("0 " + " ".join(coords))

        # For negative samples (no POI), keep an empty label file.
        lbl_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        return img_path, lbl_path

    def set_capture_for_relabel_callback(self, callback) -> None:
        self._capture_for_relabel_callback = callback

    def _capture_for_relabel(self) -> None:
        frame = self._last_infer_frame
        result = self._last_infer_result
        if frame is None or result is None:
            self.status_var.set("Nothing to capture yet. Run one prediction first.")
            return
        saved = self._save_prediction_as_label(frame, result)
        if not saved:
            self.status_var.set("Capture failed: could not save current frame.")
            return
        img_path, _ = saved
        self._autolabel_count += 1
        self._update_autolabel_counter()
        self.status_var.set(f"Captured for relabel: {img_path.name}")
        if self._capture_for_relabel_callback:
            self._capture_for_relabel_callback(img_path, self.autolabel_dir / "labels")

    def _update_autolabel_counter(self) -> None:
        if self._autolabel_counter_lbl:
            self._autolabel_counter_lbl.configure(
                text=f"\U0001F4BE {self._autolabel_count}" if self._autolabel_count else "")

    # ──────────────────────────────────────────── Output rendering ──

    def _build_outputs(
        self, bgr: np.ndarray, result,
    ) -> tuple[np.ndarray, list[dict], str]:
        overlay = bgr.copy()
        h, w = bgr.shape[:2]

        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            cv2.putText(overlay, "POI not detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            status = "POI not detected"
            if self.image_paths:
                status = f"[{self.index + 1}/{len(self.image_paths)}] {status}"
            return overlay, [], status

        all_crops: list[dict] = []
        rect_summaries: list[str] = []

        # OBB provides 4 corner points directly — no minAreaRect needed
        all_corners = obb.xyxyxyxy.cpu().numpy()  # shape (N, 4, 2)

        for i, corners in enumerate(all_corners):
            color = POI_COLORS[i % len(POI_COLORS)]
            color_hex = "#{:02x}{:02x}{:02x}".format(*color)
            poly_int = corners.reshape(-1, 1, 2).astype(np.int32)

            cv2.polylines(overlay, [poly_int], True, color, 3)

            fill_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(fill_mask, [poly_int], 255)
            color_tint = np.array(color, dtype=np.float32)
            shaded = overlay.copy()
            mask_region = fill_mask > 0
            shaded[mask_region] = (
                0.6 * shaded[mask_region].astype(np.float32) + 0.4 * color_tint
            ).astype(np.uint8)
            overlay = shaded

            rect_pts = corners.astype(np.float32)
            cx = int(np.mean(rect_pts[:, 0]))
            cy = int(np.mean(rect_pts[:, 1]))
            label_text = f"POI {i + 1}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (cx - 4, cy - th - 8), (cx + tw + 4, cy + 4), color, -1)
            cv2.putText(overlay, label_text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            quad = self._order_quad_points(rect_pts)
            crop = self._warp_quad_crop(bgr, quad)

            all_crops.append({
                "image": crop,
                "index": i + 1,
                "label": label_text,
                "color": color,
                "color_hex": color_hex,
                "bbox": (int(np.min(rect_pts[:, 0])), int(np.min(rect_pts[:, 1])),
                         int(np.max(rect_pts[:, 0])), int(np.max(rect_pts[:, 1]))),
            })
            x0, y0 = int(np.min(rect_pts[:, 0])), int(np.min(rect_pts[:, 1]))
            x1, y1 = int(np.max(rect_pts[:, 0])), int(np.max(rect_pts[:, 1]))
            rect_summaries.append(f"{i + 1}:({x0},{y0})-({x1},{y1})")

        n = len(all_crops)
        poi_word = "POI" if n == 1 else "POIs"
        rects_text = ", ".join(rect_summaries)
        status = f"{n} {poi_word} detected | {rects_text}"
        if self.image_paths:
            status = f"[{self.index + 1}/{len(self.image_paths)}] {status}"
        return overlay, all_crops, status

    # ───────────────────────────────────────────────────────── Display ──

    def _set_preview_images(self, overlay_bgr: np.ndarray,
                            crops_data: list[dict]) -> None:
        # Left: overlay
        left = self._to_tk_image(overlay_bgr, 660, 660)
        self.left_photo = left
        self.left_label.configure(image=self.left_photo)

        # Right: scrollable grid of crop cards
        # Clear old widgets
        for child in self._crops_inner.winfo_children():
            child.destroy()
        self._crop_photos.clear()

        if not crops_data:
            lbl = tk.Label(self._crops_inner, text="No POI detected",
                           bg=BG_CARD, fg=FG_DIM, font=(FONT, 11))
            lbl.grid(row=0, column=0, padx=40, pady=40)
            return

        # Configure 2-column grid
        cols = 2
        for c in range(cols):
            self._crops_inner.columnconfigure(c, weight=1, uniform="col")

        for idx, crop_info in enumerate(crops_data):
            row, col = divmod(idx, cols)
            self._add_crop_card(crop_info, row, col)

    def _add_crop_card(self, crop_info: dict, row: int, col: int) -> None:
        card = tk.Frame(self._crops_inner, bg=BG_INPUT, bd=0,
                        highlightthickness=1, highlightbackground=BG_HOVER)
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)

        # Colored header bar with POI number
        header = tk.Frame(card, bg=crop_info["color_hex"], height=24)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=f"  {crop_info['label']}", bg=crop_info["color_hex"],
                 fg="#000000", font=(FONT, 9, "bold"), anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bounding box info
        x0, y0, x1, y1 = crop_info["bbox"]
        tk.Label(header, text=f"({x0},{y0})-({x1},{y1})  ", bg=crop_info["color_hex"],
                 fg="#000000", font=(MONO, 7), anchor="e"
                 ).pack(side=tk.RIGHT)

        # Crop image
        crop_bgr = crop_info["image"]
        ch, cw = crop_bgr.shape[:2]
        # Scale to fit nicely: max 250px on either side
        target_size = 250
        scale = min(target_size / max(cw, 1), target_size / max(ch, 1), 1.0)
        display_w = max(1, int(cw * scale))
        display_h = max(1, int(ch * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop_bgr, (display_w, display_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        img_lbl = tk.Label(card, image=photo, bg=BG_CARD, bd=0)
        img_lbl.pack(padx=4, pady=(2, 4))

        # Keep reference to prevent GC
        self._crop_photos.append(photo)


    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered

    @staticmethod
    def _warp_quad_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _to_tk_image(bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=image)

    def _load_last_trained(self) -> None:
        """Load the most recent trained model from runs/ directory."""
        self._stop_live()
        try:
            candidates = list(RUNS_DIR.rglob("best.pt"))
            if not candidates:
                messagebox.showinfo("No Models", "No trained models found in runs/ directory.")
                return
            # Sort by modification time (most recent first)
            last_model = max(candidates, key=lambda p: p.stat().st_mtime)
            self.model_path = last_model
            self._load_model()
            self.status_var.set(f"Loaded last trained model: {last_model.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find last trained model: {e}")

    # ────────────────────────────────────────────────────────── Cleanup ──

    def _on_close(self) -> None:
        self._stop_live()
        self._remove_screen_exclusion()
        if not self.embedded:
            self.window.destroy()


# ── Data panel (label + dataset prep) ─────────────────────────────────────


class DataManagementTab(tk.Frame):
    """Data surface: Label tab + Dataset prep with live preview (pairs → crops → synthetics)."""

    _PATHS_JSON = ".poi_studio_dataset.json"
    _PREVIEW_MAX_W = 700
    _PREVIEW_MAX_H = 420
    _FILMSTRIP_MAX = 14
    _THUMB = 52
    # Heuristic: filenames produced by this app’s synthetic generator / extract step
    _GEN_SYNTH_NAME_RE = re.compile(r"(?i)^synth_\d")
    _GEN_EXTRACT_NAME_RE = re.compile(r"(?i)_poi_\d+$")
    # (code → listbox tag, explanation for preview / counts)
    _GEN_SIGNAL_META: dict[str, tuple[str, str]] = {
        "filename_synth": ("SYN", "Filename matches synthetic batch (synth_…)"),
        "filename_crop": ("CRP", "Filename matches extract crop (*_poi_n)"),
        "path_synthetic_output": ("OUT", "File lives under Working/data_master/synthetic_output/images"),
        "path_synthetic_assets": ("AST", "File lives under Working/data_master/synthetic_assets"),
    }

    def __init__(self, parent, main_app):
        super().__init__(parent, bg=BG)
        self.main_app = main_app
        self.work_dir = Path(".").resolve()
        self.project_path = self.work_dir
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.view_mode = "pairs"
        self._pair_list: list[tuple[Path, Path]] = []
        self._crop_paths: list[Path] = []
        self._synthetic_paths: list[Path] = []
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._filmstrip_photos: list[ImageTk.PhotoImage] = []
        self._filmstrip_paths: deque[Path] = deque(maxlen=self._FILMSTRIP_MAX)
        self._pair_thumb_photos: list[ImageTk.PhotoImage] = []
        self._generating = False
        self._tab_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._tab_ui_poll_id: str | None = None

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        label_host = tk.Frame(self.notebook, bg=BG)
        dataset_host = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(label_host, text="Label")
        self.notebook.add(dataset_host, text="Dataset")

        self.relabel = RelabelCornersTool(label_host, embedded=True, bind_shortcuts=False)
        self._build_dataset_panel(dataset_host)
        self._try_restore_paths()
        self._schedule_tab_ui_poll()
        self.pack(fill=tk.BOTH, expand=True)

    def focus_label_tab(self) -> None:
        try:
            self.notebook.select(0)
        except tk.TclError:
            pass

    def _schedule_tab_ui_poll(self) -> None:
        self._process_tab_ui_queue()
        try:
            self._tab_ui_poll_id = self.main_app.root.after(120, self._schedule_tab_ui_poll)
        except tk.TclError:
            self._tab_ui_poll_id = None

    def _process_tab_ui_queue(self) -> None:
        while True:
            try:
                item = self._tab_ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "pipeline":
                self.pipeline_status.configure(text=item[1])
            elif kind == "extract_complete":
                self._after_extract_refresh()
            elif kind == "synthetic_saved":
                self._on_synthetic_saved_ui(item[1])
            elif kind == "generate_done":
                self._generating = False

    def _build_dataset_panel(self, scroll_parent: tk.Frame) -> None:
        canvas = tk.Canvas(scroll_parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=BG)
        cw_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def _sync_scroll(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable.bind("<Configure>", _sync_scroll)

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(cw_id, width=event.width)
            _sync_scroll()

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        for w in (canvas, scrollable):
            w.bind("<MouseWheel>", _on_mousewheel)

        pad_x = 28

        hero = tk.Frame(scrollable, bg=BG)
        hero.pack(fill="x", padx=pad_x, pady=(18, 6))
        stripe = tk.Frame(hero, bg=ACCENT, height=4)
        stripe.pack(fill="x")
        hero_inner = tk.Frame(hero, bg=BG_PANEL)
        hero_inner.pack(fill="x")
        tk.Label(
            hero_inner,
            text="Dataset",
            bg=BG_PANEL,
            fg=FG,
            font=(FONT, 22, "bold"),
            anchor="w",
        ).pack(anchor="w", padx=22, pady=(18, 4))
        tk.Label(
            hero_inner,
            text="Choose image + label folders, verify pairs in the preview, then extract → generate. Path choices are saved under your working folder.",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 11),
            anchor="w",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 18))

        sec_folders = self._dataset_step_section(
            scrollable,
            1,
            "Folders and preview",
            "Pick folders with matching stems (image.jpg + image.txt). Load pairs to preview with label overlays. "
            "Likely-generated images are flagged by filename (synth_…, *_poi_n) and/or path under Working synthetic_output "
            "or synthetic_assets — see [gen:…] tags, orange thumbnails, and preview explanations.",
        )
        sec_folders.pack(fill="x", padx=pad_x, pady=(8, 6))

        flow = tk.Frame(sec_folders, bg=BG_PANEL)
        flow.pack(fill="x", padx=18, pady=(4, 8))
        FlatButton(
            flow,
            text="Images folder",
            command=self._pick_images_dir,
            bg=ACCENT,
            bg_hover=ACCENT_HOVER,
            width=118,
            height=32,
            font_size=9,
            bold=True,
        ).pack(side="left", padx=(2, 6))
        FlatButton(
            flow,
            text="Labels folder",
            command=self._pick_labels_dir,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=118,
            height=32,
            font_size=9,
        ).pack(side="left", padx=(0, 6))
        FlatButton(
            flow,
            text="Working folder",
            command=self._pick_work_dir,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=124,
            height=32,
            font_size=9,
        ).pack(side="left", padx=(0, 6))
        FlatButton(
            flow,
            text="Load pairs",
            command=self._load_pairs,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=100,
            height=32,
            font_size=9,
            bold=True,
        ).pack(side="left", padx=(8, 6))

        paths_col = tk.Frame(sec_folders, bg=BG_PANEL)
        paths_col.pack(fill="x", padx=18, pady=(0, 8))
        self.path_images_lbl = tk.Label(
            paths_col,
            text="Images: —",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_images_lbl.pack(fill="x")
        self.path_labels_lbl = tk.Label(
            paths_col,
            text="Labels: —",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_labels_lbl.pack(fill="x")
        self.path_work_lbl = tk.Label(
            paths_col,
            text=f"Working (outputs): {self.work_dir}",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_work_lbl.pack(fill="x")

        thumb_outer = tk.Frame(sec_folders, bg=BG_PANEL)
        thumb_outer.pack(fill="x", padx=18, pady=(4, 6))
        tk.Label(
            thumb_outer,
            text="Loaded pair thumbnails (click to preview)",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 8),
            anchor="w",
        ).pack(anchor="w")
        self.pair_thumb_strip = tk.Frame(thumb_outer, bg=BG_PANEL)
        self.pair_thumb_strip.pack(fill="x", pady=(4, 0))

        preview_card = tk.Frame(sec_folders, bg=BG_PANEL)
        preview_card.pack(fill="both", expand=True, padx=18, pady=(4, 12))
        preview_row = tk.Frame(preview_card, bg=BG_PANEL)
        preview_row.pack(fill="both", expand=True)

        left_list = tk.Frame(preview_row, bg=BG_PANEL)
        left_list.pack(side="left", fill="y", anchor="n")
        tk.Label(left_list, text="Items", bg=BG_PANEL, fg=FG, font=(FONT, 9, "bold")).pack(anchor="w")
        self.pair_listbox = tk.Listbox(
            left_list,
            bg=BG_INPUT,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            font=(MONO, 9),
            height=16,
            width=34,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            activestyle="none",
        )
        self.pair_listbox.pack(fill="y")
        self.pair_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        right_prev = tk.Frame(
            preview_row,
            bg=BG_INPUT,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        right_prev.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.preview_mode_lbl = tk.Label(
            right_prev,
            text="Preview: select an item",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
            justify="left",
            wraplength=880,
        )
        self.preview_mode_lbl.pack(fill="x", padx=10, pady=(8, 4))
        self.preview_canvas = tk.Canvas(
            right_prev,
            width=self._PREVIEW_MAX_W,
            height=self._PREVIEW_MAX_H,
            bg="#101010",
            highlightthickness=0,
        )
        self.preview_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        film_wrap = tk.Frame(right_prev, bg=BG_INPUT)
        film_wrap.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(
            film_wrap,
            text="Recent synthetics (latest on the right)",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(FONT, 8),
            anchor="w",
        ).pack(anchor="w")
        self.filmstrip_inner = tk.Frame(film_wrap, bg=BG_INPUT)
        self.filmstrip_inner.pack(fill="x", pady=(4, 0))

        sec2 = self._dataset_step_section(
            scrollable,
            2,
            "Build the dataset",
            "Runs are optional and ordered A → B → C. Everything is written under your Working folder (Step 1). "
            "Your original Images / Labels folders are only read by A; they are not overwritten.",
        )
        sec2.pack(fill="x", padx=pad_x, pady=(8, 6))

        hint_card = tk.Frame(sec2, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        hint_card.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(
            hint_card,
            text=(
                "Where files go (relative to Working folder)\n"
                "• A Extract → data_master/synthetic_assets/  (PNG crops per POI, alpha)\n"
                "• B Generate → data_master/synthetic_output/images/synth_*.jpg  "
                "+ matching labels in synthetic_output/labels/\n"
                "• C Split → runs the built-in reorganizer; by default it prints a dry-run unless you launch "
                "with --run (check terminal output)"
            ),
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(MONO, 8),
            anchor="w",
            justify="left",
            wraplength=920,
        ).pack(fill="x", padx=12, pady=10)

        gen_row = tk.Frame(sec2, bg=BG_PANEL)
        gen_row.pack(fill="x", padx=18, pady=(0, 8))
        tk.Label(gen_row, text="Synthetic count", bg=BG_PANEL, fg=FG, font=(FONT, 9)).pack(side="left", padx=(2, 8))
        self.synth_count_var = tk.StringVar(value="50")
        tk.Entry(
            gen_row,
            textvariable=self.synth_count_var,
            width=6,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG,
            font=(MONO, 10),
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            justify="center",
        ).pack(side="left", padx=(0, 12))

        self._pipeline_row(
            sec2,
            "A",
            "Extract assets",
            self._extract_assets,
            "Reads Images + Labels folders (same as Load pairs). Calls extract_assets: writes PNG crops to "
            "Working/data_master/synthetic_assets/. Then the preview switches to those crops.",
        )
        self._pipeline_row(
            sec2,
            "B",
            "Generate synthetic",
            self._generate_synthetic,
            "Needs: Working/data_master/synthetic_assets/ (from A) and "
            "Working/data_master/without_poi_negatives/images/ (background JPGs/PNGs). "
            "Writes synth_XXXXX.jpg + .txt under synthetic_output/. Preview jumps to each new image.",
            button_attr="btn_generate",
        )
        self._pipeline_row(
            sec2,
            "C",
            "Split train / val / test",
            self._reorganize_data,
            "Runs reorganize_dataset_main from this app (hard-coded source paths). Often dry-run only unless "
            "you started Python with --run; read the terminal. Use when your project matches that script.",
        )

        wipe_row = tk.Frame(sec2, bg=BG_PANEL)
        wipe_row.pack(fill="x", padx=18, pady=(10, 6))
        FlatButton(
            wipe_row,
            text="Clear generated outputs…",
            command=self._clear_generated_outputs_confirm,
            bg=RED,
            bg_hover=RED_HOVER,
            width=178,
            height=32,
            font_size=9,
            bold=True,
        ).pack(side="left", padx=(2, 12))
        tk.Label(
            wipe_row,
            text="Opens a review window: per-folder counts, sizes, extensions, largest files, scrollable path list, "
            "then checkbox + Delete. Only synthetic_assets and synthetic_output under Working.",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
            wraplength=720,
            justify="left",
        ).pack(side="left", fill="x", expand=True)

        status_bar = tk.Frame(sec2, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        status_bar.pack(fill="x", padx=18, pady=(8, 14))
        tk.Label(status_bar, text="Pipeline", bg=BG_INPUT, fg=ACCENT,
                 font=(FONT, 8, "bold")).pack(side="left", padx=(12, 8), pady=8)
        self.pipeline_status = tk.Label(
            status_bar,
            text="Idle — choose folders and Load pairs, then run A → B → C when ready.",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.pipeline_status.pack(side="left", fill="x", expand=True, pady=8, padx=(0, 12))

    def _pipeline_row(
        self,
        parent: tk.Misc,
        mark: str,
        button_text: str,
        command,
        description: str,
        *,
        button_attr: str | None = None,
    ) -> None:
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=18, pady=3)
        lane = tk.Frame(row, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        lane.pack(fill="x")
        badge = tk.Frame(lane, bg=BG_INPUT, width=40)
        badge.pack(side="left", fill="y")
        badge.pack_propagate(False)
        tk.Label(badge, text=mark, bg=BG_INPUT, fg=ACCENT, font=(FONT, 12, "bold")).pack(
            expand=True,
        )
        mid = tk.Frame(lane, bg=BG_INPUT)
        mid.pack(side="left", fill="y", padx=(0, 12), pady=10)
        btn = FlatButton(
            mid,
            text=button_text,
            command=command,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=168,
            height=30,
            font_size=9,
        )
        btn.pack(anchor="w")
        if button_attr:
            setattr(self, button_attr, btn)
        tk.Label(
            lane,
            text=description,
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
            wraplength=720,
            justify="left",
        ).pack(side="left", fill="x", expand=True, pady=10, padx=(0, 14))

    def _dataset_step_section(self, parent, step_num: int, title: str, description: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=BG)
        card = tk.Frame(
            outer,
            bg=BG_CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.pack(fill="x")

        head = tk.Frame(card, bg=BG_CARD)
        head.pack(fill="x", padx=0, pady=(16, 8))
        rail = tk.Frame(head, bg=ACCENT, width=4)
        rail.pack(side="left", fill="y", padx=(16, 0))
        rail.pack_propagate(False)

        tit_col = tk.Frame(head, bg=BG_CARD)
        tit_col.pack(side="left", fill="x", expand=True, padx=(14, 16))
        tk.Label(
            tit_col,
            text=f"STEP {step_num}",
            bg=BG_CARD,
            fg=ACCENT,
            font=(FONT, 8, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            tit_col,
            text=title,
            bg=BG_CARD,
            fg=FG,
            font=(FONT, 13, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        tk.Label(
            card,
            text=description,
            bg=BG_CARD,
            fg=FG_DIM,
            font=(FONT, 10),
            anchor="w",
            wraplength=880,
            justify="left",
        ).pack(fill="x", padx=(34, 22), pady=(0, 12))

        return outer

    # ── Paths / persistence ───────────────────────────────────────────────

    def _paths_json_file(self) -> Path:
        return self.work_dir / self._PATHS_JSON

    def _save_paths_json(self) -> None:
        try:
            payload = {
                "images_dir": str(self.images_dir) if self.images_dir else "",
                "labels_dir": str(self.labels_dir) if self.labels_dir else "",
                "work_dir": str(self.work_dir),
            }
            self._paths_json_file().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _try_restore_paths(self) -> None:
        for candidate in (self.work_dir, Path(".").resolve()):
            j = candidate / self._PATHS_JSON
            if not j.is_file():
                continue
            try:
                data = json.loads(j.read_text(encoding="utf-8"))
                wd = Path(data.get("work_dir", "") or candidate)
                self.work_dir = wd.resolve()
                self.project_path = self.work_dir
                img = data.get("images_dir") or ""
                lab = data.get("labels_dir") or ""
                self.images_dir = Path(img).resolve() if img else None
                self.labels_dir = Path(lab).resolve() if lab else None
                self.path_work_lbl.configure(text=f"Working (outputs): {self.work_dir}")
                if self.images_dir:
                    self.path_images_lbl.configure(text=f"Images: {self.images_dir}")
                if self.labels_dir:
                    self.path_labels_lbl.configure(text=f"Labels: {self.labels_dir}")
                break
            except (json.JSONDecodeError, OSError, ValueError):
                continue

    def _pick_images_dir(self) -> None:
        d = filedialog.askdirectory(title="Folder containing training images")
        if not d:
            return
        self.images_dir = Path(d).resolve()
        self.path_images_lbl.configure(text=f"Images: {self.images_dir}")
        self._save_paths_json()

    def _pick_labels_dir(self) -> None:
        d = filedialog.askdirectory(title="Folder containing YOLO label .txt files")
        if not d:
            return
        self.labels_dir = Path(d).resolve()
        self.path_labels_lbl.configure(text=f"Labels: {self.labels_dir}")
        self._save_paths_json()

    def _pick_work_dir(self) -> None:
        d = filedialog.askdirectory(title="Working folder (data_master/ outputs go here)")
        if not d:
            return
        self.work_dir = Path(d).resolve()
        self.project_path = self.work_dir
        self.path_work_lbl.configure(text=f"Working (outputs): {self.work_dir}")
        self._save_paths_json()

    def _assets_output_dir(self) -> Path:
        return self.work_dir / "data_master" / "synthetic_assets"

    def _synthetic_images_dir(self) -> Path:
        return self.work_dir / "data_master" / "synthetic_output" / "images"

    def _synthetic_labels_dir(self) -> Path:
        return self.work_dir / "data_master" / "synthetic_output" / "labels"

    @staticmethod
    def _format_byte_size(n: int) -> str:
        if n < 1024:
            return f"{n} B"
        if n < 1024 * 1024:
            return f"{n / 1024:.1f} KiB"
        return f"{n / (1024 * 1024):.2f} MiB"

    @staticmethod
    def _safe_file_size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    def _scan_generated_files(self) -> tuple[dict[str, list[Path]], int]:
        """Collect files under pipeline output dirs only (under Working folder)."""
        work = self.work_dir.resolve()
        roots: dict[str, Path] = {
            "data_master/synthetic_assets": self._assets_output_dir(),
            "data_master/synthetic_output/images": self._synthetic_images_dir(),
            "data_master/synthetic_output/labels": self._synthetic_labels_dir(),
        }
        buckets: dict[str, list[Path]] = {k: [] for k in roots}
        total_bytes = 0
        for key, root in roots.items():
            r = root.resolve()
            if not r.is_dir():
                continue
            try:
                r.relative_to(work)
            except ValueError:
                continue
            for p in r.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    p.resolve().relative_to(work)
                except ValueError:
                    continue
                buckets[key].append(p)
                total_bytes += self._safe_file_size(p)
        return buckets, total_bytes

    @staticmethod
    def _extension_histogram(paths: list[Path]) -> dict[str, int]:
        c: dict[str, int] = {}
        for p in paths:
            ext = p.suffix.lower() if p.suffix else "(no extension)"
            c[ext] = c.get(ext, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    def _rel_to_work(self, path: Path, work: Path) -> str:
        try:
            return str(path.resolve().relative_to(work))
        except ValueError:
            return str(path.resolve())

    def _execute_generated_delete(self, buckets: dict[str, list[Path]], total_files: int) -> None:
        deleted = 0
        errors: list[str] = []
        for lst in buckets.values():
            for p in lst:
                try:
                    p.unlink()
                    deleted += 1
                except OSError as e:
                    errors.append(f"{p}: {e}")

        self._filmstrip_paths.clear()
        self._refresh_filmstrip()
        self._crop_paths.clear()
        self._synthetic_paths.clear()

        if self.view_mode in ("crops", "synthetic"):
            self.view_mode = "pairs"
            self._fill_listbox_for_mode()
            self._rebuild_pair_thumbnails()
            self.preview_canvas.delete("all")
            if self._pair_list:
                self.preview_mode_lbl.configure(text=f"Preview: source pairs ({len(self._pair_list)} loaded)")
                self.pair_listbox.selection_clear(0, "end")
                self.pair_listbox.selection_set(0)
                self.pair_listbox.see(0)
                self._show_pair_at_index(0)
            else:
                self.preview_mode_lbl.configure(text="Preview: cleared — load pairs to continue")

        summary = f"Removed {deleted} of {total_files} file(s)."
        self.pipeline_status.configure(text=summary)
        if errors:
            err_txt = "\n".join(errors[:15])
            if len(errors) > 15:
                err_txt += f"\n… and {len(errors) - 15} more error(s)"
            summary += f"\n\nErrors ({len(errors)}):\n{err_txt}"
        messagebox.showinfo("Clear generated outputs", summary)

    def _open_clear_generated_review(self, buckets: dict[str, list[Path]], total_bytes: int) -> None:
        """Modal safety review: counts, extensions, largest files, path preview, checkbox, then delete."""
        total_files = sum(len(v) for v in buckets.values())
        work = self.work_dir.resolve()

        top = tk.Toplevel(self.main_app.root)
        top.title("Review deletion — generated outputs")
        top.configure(bg=BG)
        top.transient(self.main_app.root)
        top.grab_set()
        top.minsize(620, 520)
        px = self.main_app.root.winfo_rootx() + 40
        py = self.main_app.root.winfo_rooty() + 40
        top.geometry(f"760x580+{px}+{py}")

        header = tk.Frame(top, bg=BG_PANEL, padx=14, pady=12)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="You are about to permanently delete generated pipeline files.",
            bg=BG_PANEL,
            fg=FG,
            font=(FONT, 12, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Only the folders below (under your Working folder) are scanned. Nothing else on disk is touched.",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        tk.Label(
            header,
            text=f"Working folder:\n  {work}",
            bg=BG_PANEL,
            fg=ACCENT,
            font=(MONO, 9),
            anchor="w",
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        body = tk.Frame(top, bg=BG, padx=14, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        sum_lines: list[str] = [
            "SUMMARY",
            f"  Total files matched: {total_files}",
            f"  Total size (approx.): {self._format_byte_size(total_bytes)}",
            "",
            "PER FOLDER:",
        ]
        all_paths: list[Path] = []
        for key in sorted(buckets.keys()):
            lst = buckets[key]
            sz = sum(self._safe_file_size(p) for p in lst)
            sum_lines.append(f"  • {key}")
            sum_lines.append(f"      files: {len(lst)}   size: ~{self._format_byte_size(sz)}")
            hist = self._extension_histogram(lst)
            if hist:
                parts = [f"{ext}: {n}" for ext, n in hist.items()]
                sum_lines.append("      extensions: " + ", ".join(parts))
            sum_lines.append("")
            all_paths.extend(lst)

        sum_lines.extend(
            [
                "EXCLUDED (never deleted by this action):",
                "  • Your Images folder and Labels folder from Step 1",
                "  • data_master/labeled_with_poi/ …",
                "  • data_master/without_poi_negatives/",
                "  • Training runs (runs/), reorganized data unless under paths above",
                "",
                "LARGEST FILES (top 15 by size):",
            ]
        )
        ranked = sorted(all_paths, key=self._safe_file_size, reverse=True)
        for p in ranked[:15]:
            sz = self._safe_file_size(p)
            sum_lines.append(f"  {self._format_byte_size(sz):>12}  {self._rel_to_work(p, work)}")
        if len(ranked) > 15:
            sum_lines.append(f"  … {len(ranked) - 15} more file(s) not listed here")

        sum_txt = tk.Text(
            body,
            height=14,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG,
            font=(MONO, 9),
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            wrap="word",
        )
        sum_txt.pack(fill=tk.X, pady=(0, 8))
        sum_txt.insert("1.0", "\n".join(sum_lines))
        sum_txt.configure(state="disabled")

        tk.Label(
            body,
            text=f"Path preview (relative to Working folder) — first {min(400, total_files)} of {total_files}:",
            bg=BG,
            fg=FG_DIM,
            font=(FONT, 9, "bold"),
            anchor="w",
        ).pack(anchor="w")

        preview_wrap = tk.Frame(body, bg=BORDER)
        preview_wrap.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        preview_txt = tk.Text(
            preview_wrap,
            bg=BG_CARD,
            fg=FG,
            insertbackground=FG,
            font=(MONO, 8),
            bd=0,
            highlightthickness=0,
            wrap="none",
        )
        sy = ttk.Scrollbar(preview_wrap, orient=tk.VERTICAL, command=preview_txt.yview)
        sx = ttk.Scrollbar(preview_wrap, orient=tk.HORIZONTAL, command=preview_txt.xview)
        preview_txt.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        preview_txt.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        preview_wrap.grid_rowconfigure(0, weight=1)
        preview_wrap.grid_columnconfigure(0, weight=1)

        preview_lines: list[str] = []
        sorted_paths = sorted(all_paths, key=lambda p: self._rel_to_work(p, work).lower())
        preview_cap = 400
        for i, p in enumerate(sorted_paths[:preview_cap]):
            preview_lines.append(self._rel_to_work(p, work))
        if total_files > preview_cap:
            preview_lines.append("")
            preview_lines.append(f"... {total_files - preview_cap} more paths not shown (still included in delete count).")

        preview_txt.insert("1.0", "\n".join(preview_lines))
        preview_txt.configure(state="disabled")

        ack = tk.BooleanVar(value=False)

        foot = tk.Frame(top, bg=BG_PANEL, padx=14, pady=12)
        foot.pack(fill=tk.X, side=tk.BOTTOM)

        def sync_delete_btn(*_a) -> None:
            delete_btn.configure(state="normal" if ack.get() else "disabled")

        chk = tk.Checkbutton(
            foot,
            text=f"I understand this will permanently delete {total_files} file(s) (~{self._format_byte_size(total_bytes)}) from the folders above. "
            "This cannot be undone.",
            variable=ack,
            bg=BG_PANEL,
            fg=FG,
            font=(FONT, 10),
            activebackground=BG_PANEL,
            selectcolor=BG_INPUT,
            wraplength=680,
            anchor="w",
            justify="left",
            command=sync_delete_btn,
        )
        chk.pack(anchor="w", pady=(0, 10))

        btns = tk.Frame(foot, bg=BG_PANEL)
        btns.pack(fill=tk.X)

        def on_cancel() -> None:
            top.grab_release()
            top.destroy()

        delete_btn = tk.Button(
            btns,
            text=f"Delete {total_files} file(s)",
            state="disabled",
            bg=RED,
            fg="#ffffff",
            activebackground=RED_HOVER,
            activeforeground="#ffffff",
            font=(FONT, 10, "bold"),
            padx=16,
            pady=8,
            cursor="hand2",
            command=lambda: self._on_clear_generated_delete_confirmed(top, buckets, total_files),
        )
        delete_btn.pack(side=tk.RIGHT, padx=(8, 0))

        tk.Button(
            btns,
            text="Cancel",
            bg=BG_CARD,
            fg=FG,
            activebackground=BG_HOVER,
            font=(FONT, 10),
            padx=16,
            pady=8,
            command=on_cancel,
        ).pack(side=tk.RIGHT)

        top.protocol("WM_DELETE_WINDOW", on_cancel)

    def _on_clear_generated_delete_confirmed(
        self,
        dialog: tk.Toplevel,
        buckets: dict[str, list[Path]],
        total_files: int,
    ) -> None:
        dialog.grab_release()
        dialog.destroy()
        self.main_app.root.update_idletasks()
        self._execute_generated_delete(buckets, total_files)

    def _clear_generated_outputs_confirm(self) -> None:
        buckets, total_bytes = self._scan_generated_files()
        total_files = sum(len(v) for v in buckets.values())
        if total_files == 0:
            messagebox.showinfo(
                "Nothing to delete",
                "No generated files were found under your Working folder in:\n\n"
                "• data_master/synthetic_assets/\n"
                "• data_master/synthetic_output/images/\n"
                "• data_master/synthetic_output/labels/\n\n"
                "Use “Working folder” to point at the project that contains those folders.",
            )
            return

        self._open_clear_generated_review(buckets, total_bytes)

    def _collect_pairs(self) -> list[tuple[Path, Path]]:
        """Match images under images_dir (recursive) to labels under labels_dir by stem (case-insensitive)."""
        assert self.images_dir is not None and self.labels_dir is not None
        label_by_stem: dict[str, Path] = {}
        for lab_path in self.labels_dir.rglob("*.txt"):
            if lab_path.is_file():
                label_by_stem[lab_path.stem.lower()] = lab_path
        pairs: list[tuple[Path, Path]] = []
        for img_path in self.images_dir.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            lab = label_by_stem.get(img_path.stem.lower())
            if lab is not None:
                pairs.append((img_path, lab))
        pairs.sort(key=lambda t: str(t[0]).lower())
        return pairs

    def _labeled_pair_generation_signals(self, image_path: Path) -> list[tuple[str, str]]:
        """Heuristic signals that an image is pipeline-generated (not pixel-perfect proof).

        Uses (1) filename patterns from this app and (2) location under the current Working folder.
        """
        sigs: list[tuple[str, str]] = []
        stem = image_path.stem
        if self._GEN_SYNTH_NAME_RE.match(stem):
            meta = self._GEN_SIGNAL_META["filename_synth"]
            sigs.append(("filename_synth", meta[1]))
        if self._GEN_EXTRACT_NAME_RE.search(stem):
            meta = self._GEN_SIGNAL_META["filename_crop"]
            sigs.append(("filename_crop", meta[1]))
        try:
            rp = image_path.resolve()
            rw = self.work_dir.resolve()
            out_root = (rw / "data_master" / "synthetic_output" / "images").resolve()
            ast_root = (rw / "data_master" / "synthetic_assets").resolve()
            if out_root.is_dir():
                try:
                    rp.relative_to(out_root)
                    meta = self._GEN_SIGNAL_META["path_synthetic_output"]
                    sigs.append(("path_synthetic_output", meta[1]))
                except ValueError:
                    pass
            if ast_root.is_dir():
                try:
                    rp.relative_to(ast_root)
                    meta = self._GEN_SIGNAL_META["path_synthetic_assets"]
                    sigs.append(("path_synthetic_assets", meta[1]))
                except ValueError:
                    pass
        except OSError:
            pass
        dedup: list[tuple[str, str]] = []
        seen_codes: set[str] = set()
        for code, msg in sigs:
            if code in seen_codes:
                continue
            seen_codes.add(code)
            dedup.append((code, msg))
        return dedup

    def _labeled_pair_has_generation_signals(self, image_path: Path) -> bool:
        return bool(self._labeled_pair_generation_signals(image_path))

    @staticmethod
    def _imread_bgr(path: Path) -> np.ndarray | None:
        """cv2.imread with Unicode path fallback on Windows."""
        p = str(path)
        img = cv2.imread(p)
        if img is not None:
            return img
        try:
            buf = np.fromfile(p, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _load_pairs(self) -> None:
        if not self.images_dir or not self.labels_dir:
            messagebox.showwarning("Folders", "Choose both Images folder and Labels folder first.")
            return
        pairs = self._collect_pairs()
        self._pair_list = pairs
        self.view_mode = "pairs"
        self._fill_listbox_for_mode()
        self._rebuild_pair_thumbnails()
        n = len(pairs)
        flagged = 0
        sig_counter: Counter[str] = Counter()
        for img_p, _ in pairs:
            ss = self._labeled_pair_generation_signals(img_p)
            if ss:
                flagged += 1
            for code, _msg in ss:
                sig_counter[code] += 1
        self.preview_mode_lbl.configure(text=f"Preview: source pairs ({n} loaded)")
        status = f"Loaded {n} pair(s)."
        if flagged:
            order = ("filename_synth", "filename_crop", "path_synthetic_output", "path_synthetic_assets")
            bits = [
                f"{flagged} likely-generated (by name/path)",
            ]
            for code in order:
                k = sig_counter.get(code, 0)
                if k:
                    tag = self._GEN_SIGNAL_META[code][0]
                    bits.append(f"{tag}×{k}")
            status += " " + ", ".join(bits) + ". See [gen:TAGS] in list + preview text."
        else:
            status += (
                " No pipeline filename/path signals vs Working folder — "
                "could still be renamed synthetics."
            )
        self.pipeline_status.configure(text=status)
        self._save_paths_json()
        if n == 0:
            messagebox.showinfo(
                "No pairs found",
                "No matching image + .txt label pairs were found.\n\n"
                "Check that:\n"
                "• Each image (jpg/png/…) has a same-name .txt in the labels folder (e.g. photo.jpg + photo.txt)\n"
                "• Filenames match even in subfolders (labels are found recursively)\n"
                "• File extensions are supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
            return
        self.pair_listbox.selection_clear(0, "end")
        self.pair_listbox.selection_set(0)
        self.pair_listbox.see(0)
        self.update_idletasks()
        self._show_pair_at_index(0)

    def _fill_listbox_for_mode(self) -> None:
        self.pair_listbox.delete(0, "end")
        if self.view_mode == "pairs":
            for img_p, _ in self._pair_list:
                disp = img_p.name
                sigs = self._labeled_pair_generation_signals(img_p)
                if sigs:
                    tags = "+".join(self._GEN_SIGNAL_META[c][0] for c, _ in sigs)
                    disp = f"[gen:{tags}] " + disp
                self.pair_listbox.insert("end", disp)
        elif self.view_mode == "crops":
            for p in self._crop_paths:
                self.pair_listbox.insert("end", p.name)
        else:
            for p in self._synthetic_paths:
                self.pair_listbox.insert("end", p.name)

    def _show_pair_at_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._pair_list):
            return
        img_p, lab_p = self._pair_list[idx]
        bgr = self._imread_bgr(img_p)
        if bgr is None:
            self.preview_mode_lbl.configure(text=f"Preview: could not read image — {img_p.name}")
            self.pipeline_status.configure(text=f"Failed to open image: {img_p}")
            return
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        bgr = self._overlay_polylines(bgr, lab_p)
        self._show_bgr_on_canvas(bgr)
        sigs = self._labeled_pair_generation_signals(img_p)
        hint = ""
        if sigs:
            hint = " — " + " · ".join(msg for _c, msg in sigs)
        self.preview_mode_lbl.configure(
            text=f"Preview: {img_p.name} (+ labels){hint}",
        )

    def _rebuild_pair_thumbnails(self) -> None:
        for w in self.pair_thumb_strip.winfo_children():
            w.destroy()
        self._pair_thumb_photos.clear()
        thumb_px = 72
        max_thumbs = 40
        for idx, (img_p, _) in enumerate(self._pair_list[:max_thumbs]):
            pil_im = None
            try:
                pil_im = Image.open(img_p).convert("RGB")
                pil_im.thumbnail((thumb_px, thumb_px), Image.Resampling.LANCZOS)
            except OSError:
                bgr = self._imread_bgr(img_p)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pil_im = Image.fromarray(rgb)
                    pil_im.thumbnail((thumb_px, thumb_px), Image.Resampling.LANCZOS)
            if pil_im is None:
                continue
            ph = ImageTk.PhotoImage(pil_im)
            self._pair_thumb_photos.append(ph)

            def _click(_e: tk.Event, i: int = idx) -> None:
                self.pair_listbox.selection_clear(0, "end")
                self.pair_listbox.selection_set(i)
                self.pair_listbox.see(i)
                self._show_pair_at_index(i)

            rim = ORANGE if self._labeled_pair_has_generation_signals(img_p) else BORDER
            border = tk.Frame(self.pair_thumb_strip, bg=rim, padx=2, pady=2)
            border.pack(side="left", padx=3, pady=2)
            lbl = tk.Label(border, image=ph, bg=BG_INPUT, cursor="hand2")
            lbl.pack()
            lbl.bind("<Button-1>", _click)

        if len(self._pair_list) > max_thumbs:
            tk.Label(
                self.pair_thumb_strip,
                text=f"+ {len(self._pair_list) - max_thumbs} more (see list)",
                bg=BG_PANEL,
                fg=FG_DIM,
                font=(FONT, 8),
            ).pack(side="left", padx=8)

    def _on_list_select(self, _evt=None) -> None:
        sel = self.pair_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if self.view_mode == "pairs":
            self._show_pair_at_index(idx)
        elif self.view_mode == "crops":
            if idx >= len(self._crop_paths):
                return
            self._show_image_file_on_canvas(self._crop_paths[idx])
        else:
            if idx >= len(self._synthetic_paths):
                return
            self._show_image_file_on_canvas(self._synthetic_paths[idx])

    def _overlay_polylines(self, bgr: np.ndarray, label_path: Path) -> np.ndarray:
        parse_fn = globals().get("_parse_label")
        if parse_fn is None:
            return bgr
        try:
            text = label_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return bgr
        polys = parse_fn(text)
        if not polys:
            return bgr
        out = bgr.copy()
        h, w = out.shape[:2]
        for poly in polys:
            pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        return out

    def _show_bgr_on_canvas(self, bgr: np.ndarray) -> None:
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        tw, th = self._PREVIEW_MAX_W, self._PREVIEW_MAX_H
        scale = min(tw / max(w, 1), th / max(h, 1), 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        self._preview_photo = ImageTk.PhotoImage(im)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            tw // 2,
            th // 2,
            image=self._preview_photo,
            anchor="center",
        )

    def _show_image_file_on_canvas(self, path: Path) -> None:
        ext = path.suffix.lower()
        if ext == ".png":
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return
            if len(arr.shape) == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                self._show_bgr_on_canvas(bgr)
                return
            if arr.shape[2] == 4:
                bg = np.full((arr.shape[0], arr.shape[1], 3), 40, dtype=np.uint8)
                a = arr[:, :, 3:4].astype(np.float32) / 255.0
                bgr_f = arr[:, :, :3].astype(np.float32)
                bg_f = bg.astype(np.float32)
                comp = (bgr_f * a + bg_f * (1.0 - a)).astype(np.uint8)
                self._show_bgr_on_canvas(comp)
            else:
                self._show_bgr_on_canvas(arr)
            return
        bgr = self._imread_bgr(path)
        if bgr is not None:
            self._show_bgr_on_canvas(bgr)

    def _after_extract_refresh(self) -> None:
        out = self._assets_output_dir()
        paths = sorted(out.glob("*.png")) if out.exists() else []
        self._crop_paths = paths
        self.view_mode = "crops"
        self._fill_listbox_for_mode()
        self.preview_mode_lbl.configure(text=f"Preview: extracted crops ({len(paths)})")
        if paths:
            self.pair_listbox.selection_clear(0, "end")
            self.pair_listbox.selection_set(len(paths) - 1)
            self.pair_listbox.see(len(paths) - 1)
            self.update_idletasks()
            self._on_list_select()

    def _on_synthetic_saved_ui(self, path: Path) -> None:
        self.view_mode = "synthetic"
        d = self._synthetic_images_dir()
        self._synthetic_paths = sorted(d.glob("synth_*.jpg")) if d.exists() else []
        self._fill_listbox_for_mode()
        self.preview_mode_lbl.configure(text=f"Preview: synthetic (latest {path.name})")
        try:
            idx = self._synthetic_paths.index(path)
        except ValueError:
            idx = len(self._synthetic_paths) - 1
        if idx >= 0 and self._synthetic_paths:
            self.pair_listbox.selection_clear(0, "end")
            self.pair_listbox.selection_set(idx)
            self.pair_listbox.see(idx)
        self._show_image_file_on_canvas(path)
        self._filmstrip_paths.append(path)
        self._refresh_filmstrip()

    def _refresh_filmstrip(self) -> None:
        for ch in self.filmstrip_inner.winfo_children():
            ch.destroy()
        self._filmstrip_photos.clear()
        for p in list(self._filmstrip_paths):
            thumb = self._make_thumbnail(p, self._THUMB)
            if thumb is None:
                continue
            ph = ImageTk.PhotoImage(thumb)
            self._filmstrip_photos.append(ph)
            lbl = tk.Label(self.filmstrip_inner, image=ph, bg=BG_INPUT)
            lbl.pack(side="left", padx=2)

    def _make_thumbnail(self, path: Path, size: int) -> Image.Image | None:
        try:
            im = Image.open(path).convert("RGBA")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            return im.convert("RGB")
        except OSError:
            return None

    def _extract_assets(self) -> None:
        if not self.images_dir or not self.labels_dir:
            messagebox.showwarning("Folders", "Choose Images and Labels folders, then Load pairs.")
            return
        self.pipeline_status.configure(text="Extracting assets...")
        self.update_idletasks()
        images_dir = self.images_dir
        labels_dir = self.labels_dir
        output_dir = self._assets_output_dir()

        def _worker():
            try:
                mod = sys.modules[self.__class__.__module__]
                extract_fn = getattr(mod, "extract_assets", None)
                if extract_fn is None or not callable(extract_fn):
                    raise RuntimeError("extract_assets() missing from application module.")
                output_dir.mkdir(parents=True, exist_ok=True)
                stats = extract_fn(
                    images_dirs=[images_dir],
                    labels_dirs=[labels_dir],
                    output_dir=output_dir,
                )
                n = stats.get("extracted", 0) if isinstance(stats, dict) else 0
                self._tab_ui_queue.put(
                    ("pipeline", f"✅ Extract done — {n} asset(s) → {output_dir.name}/"),
                )
                self._tab_ui_queue.put(("extract_complete",))
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Extract error: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _generate_synthetic(self) -> None:
        if self._generating:
            return
        try:
            n = int(self.synth_count_var.get().strip())
            if n < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid count", "Enter a positive integer for synthetic count.")
            return
        self._generating = True
        self.pipeline_status.configure(text="Generating synthetic images...")
        self.update_idletasks()
        self._filmstrip_paths.clear()
        self._refresh_filmstrip()
        base = self.work_dir
        mod = sys.modules[__name__]

        def _worker():
            try:
                gen_fn = getattr(mod, "generate", None)
                if gen_fn is None:
                    raise RuntimeError("Synthetic generator not loaded in this build.")
                mod.BG_DIR = base / "data_master/without_poi_negatives/images"
                mod.ASSETS_DIR = base / "data_master/synthetic_assets"
                mod.OUT_IMG_DIR = base / "data_master/synthetic_output/images"
                mod.OUT_LBL_DIR = base / "data_master/synthetic_output/labels"
                mod.OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
                mod.OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

                def _on_saved(p: Path) -> None:
                    self._tab_ui_queue.put(("synthetic_saved", p))

                gen_fn(num_images=n, on_saved=_on_saved)
                self._tab_ui_queue.put(("pipeline", "✅ Synthetic batch finished."))
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Error: {e}"))
            finally:
                self._tab_ui_queue.put(("generate_done",))

        threading.Thread(target=_worker, daemon=True).start()

    def _reorganize_data(self) -> None:
        self.pipeline_status.configure(text="Reorganizing dataset...")
        self.update_idletasks()

        def _worker():
            try:
                mod = sys.modules[__name__]
                main_fn = getattr(mod, "reorganize_dataset_main", None)
                if main_fn is None:
                    raise RuntimeError("reorganize_dataset_main not available.")
                main_fn()
                self._tab_ui_queue.put(
                    ("pipeline", "✅ Dataset reorganized (see console / data/images/)."))
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Error: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_close(self) -> None:
        if self._tab_ui_poll_id is not None:
            try:
                self.main_app.root.after_cancel(self._tab_ui_poll_id)
            except tk.TclError:
                pass
            self._tab_ui_poll_id = None


# ── Training panel ────────────────────────────────────────────────────────


class TrainingTab(tk.Frame):
    """Train tab: weights, hyperparameters, training, validate/export."""

    def __init__(self, parent, main_app):
        super().__init__(parent, bg=BG)
        self.main_app = main_app
        self.training = False
        self.training_thread = None
        self.current_model_path = None
        self.training_stop_event = threading.Event()
        self._tab_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._tab_ui_poll_id: str | None = None

        self._build_ui()
        self._schedule_tab_ui_poll()
        # Fill the Train tab container (same pattern as viewer packing into its tab frame).
        self.pack(fill=tk.BOTH, expand=True)

    def _schedule_tab_ui_poll(self) -> None:
        self._process_tab_ui_queue()
        try:
            self._tab_ui_poll_id = self.main_app.root.after(200, self._schedule_tab_ui_poll)
        except tk.TclError:
            self._tab_ui_poll_id = None

    def _process_tab_ui_queue(self) -> None:
        while True:
            try:
                item = self._tab_ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "train_status":
                self.train_status.configure(text=item[1])
            elif kind == "train_complete":
                self._training_complete(item[1], item[2])

    # ── UI builder ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        train_pb = ttk.Style()
        try:
            train_pb.theme_use("clam")
        except tk.TclError:
            pass
        train_pb.configure(
            "PoiTrain.Horizontal.TProgressbar",
            troughcolor=BG_INPUT,
            background=ACCENT,
            darkcolor=ACCENT,
            lightcolor=ACCENT,
            bordercolor=BORDER,
            borderwidth=0,
            thickness=12,
        )

        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=BG)
        cw_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def _sync_train_scroll(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable.bind("<Configure>", _sync_train_scroll)

        def _on_train_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(cw_id, width=event.width)
            _sync_train_scroll()

        canvas.bind("<Configure>", _on_train_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        for w in (canvas, scrollable):
            w.bind("<MouseWheel>", _on_mousewheel)

        pad_x = 28

        # ── Hero ─────────────────────────────────────────────────
        hero = tk.Frame(scrollable, bg=BG)
        hero.pack(fill="x", padx=pad_x, pady=(18, 6))
        stripe = tk.Frame(hero, bg=ACCENT, height=4)
        stripe.pack(fill="x")
        hero_inner = tk.Frame(hero, bg=BG_PANEL)
        hero_inner.pack(fill="x")
        tk.Label(
            hero_inner,
            text="Train & models",
            bg=BG_PANEL,
            fg=FG,
            font=(FONT, 22, "bold"),
            anchor="w",
        ).pack(anchor="w", padx=22, pady=(18, 4))
        tk.Label(
            hero_inner,
            text="Prepare data on the Data tab, then pick weights, tune hyperparameters, and train.",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 11),
            anchor="w",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 18))

        # ── STEP 1 ───────────────────────────────────────────────
        sec3 = self._step_section(
            scrollable,
            1,
            "Weights",
            "Start from Ultralytics defaults, fine-tune a previous run, or pick a custom .pt file.",
        )
        sec3.pack(fill="x", padx=pad_x, pady=(8, 6))

        row3a = tk.Frame(sec3, bg=BG_PANEL)
        row3a.pack(fill="x", padx=18, pady=(4, 12))
        self.model_type_var = tk.StringVar(value="New Model (yolov8s-obb)")
        FlatOptionMenu(
            row3a,
            self.model_type_var,
            "New Model (yolov8s-obb)",
            "Fine-tune v1 (works with conf=0.25)",
            "Custom weights...",
            width=30,
        ).pack(side="left", padx=(2, 10))
        FlatButton(
            row3a,
            text="Browse…",
            command=self._browse_model,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=88,
            height=30,
            font_size=9,
        ).pack(side="left", padx=(0, 12))
        self.model_path_lbl = tk.Label(
            row3a,
            text="Using: yolov8s-obb.pt",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.model_path_lbl.pack(side="left", fill="x", expand=True)

        # ── STEP 2 ───────────────────────────────────────────────
        sec4 = self._step_section(
            scrollable,
            2,
            "Hyperparameters",
            "Adjust if you hit OOM, want faster iteration, or need longer runs.",
        )
        sec4.pack(fill="x", padx=pad_x, pady=(8, 6))

        hp_wrap = tk.Frame(sec4, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        hp_wrap.pack(fill="x", padx=18, pady=(4, 12))

        hp_help = {
            "Learning rate": "Typical 0.001–0.01; lower is steadier.",
            "Batch size": "Raise if VRAM allows; lower if CUDA OOM.",
            "Epochs": "More passes — longer runs.",
            "Image size": "Larger = sharper features, more VRAM.",
            "Patience": "Early-stop if val plateaus.",
        }
        defaults = {
            "Learning rate": DEFAULT_LR,
            "Batch size": DEFAULT_BATCH,
            "Epochs": DEFAULT_EPOCHS,
            "Image size": DEFAULT_IMGSZ,
            "Patience": DEFAULT_PATIENCE,
        }
        var_map = {
            "Learning rate": "lr_var",
            "Batch size": "batch_var",
            "Epochs": "epochs_var",
            "Image size": "imgsz_var",
            "Patience": "patience_var",
        }

        for i, (name, hint) in enumerate(hp_help.items()):
            row_hp = tk.Frame(hp_wrap, bg=BG_INPUT)
            row_hp.pack(fill="x", padx=14, pady=(10 if i == 0 else 6, 10 if i == len(hp_help) - 1 else 0))
            left = tk.Frame(row_hp, bg=BG_INPUT)
            left.pack(side="left", fill="y")
            tk.Label(left, text=name, bg=BG_INPUT, fg=FG, font=(FONT, 10, "bold")).pack(anchor="w")
            tk.Label(left, text=hint, bg=BG_INPUT, fg=FG_DIM, font=(FONT, 8)).pack(anchor="w")
            var = tk.StringVar(value=str(defaults[name]))
            setattr(self, var_map[name], var)
            tk.Entry(
                row_hp,
                textvariable=var,
                width=11,
                bg=BG_PANEL,
                fg=FG,
                insertbackground=FG,
                font=(MONO, 10),
                bd=0,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=ACCENT,
                justify="right",
            ).pack(side="right", padx=(12, 0))

        # ── STEP 3 ───────────────────────────────────────────────
        sec5 = self._step_section(
            scrollable,
            3,
            "Run training",
            "Runs in a background thread — you can use other tabs while epochs advance.",
        )
        sec5.pack(fill="x", padx=pad_x, pady=(8, 6))

        row5a = tk.Frame(sec5, bg=BG_PANEL)
        row5a.pack(fill="x", padx=18, pady=(4, 8))
        self.start_btn = FlatButton(
            row5a,
            text="Start training",
            command=self._start_training,
            bg=ACCENT,
            bg_hover=ACCENT_HOVER,
            width=148,
            height=38,
            font_size=10,
            bold=True,
        )
        self.start_btn.pack(side="left", padx=(2, 10))
        self.stop_btn = FlatButton(
            row5a,
            text="Stop",
            command=self._stop_training,
            bg=RED,
            bg_hover=RED_HOVER,
            width=88,
            height=38,
            font_size=10,
            bold=True,
        )
        self.stop_btn.pack(side="left", padx=(0, 14))
        self.stop_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            row5a,
            variable=self.progress_var,
            maximum=100,
            length=240,
            mode="determinate",
            style="PoiTrain.Horizontal.TProgressbar",
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 4))

        train_strip = tk.Frame(sec5, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        train_strip.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(train_strip, text="Status", bg=BG_INPUT, fg=ACCENT,
                 font=(FONT, 8, "bold")).pack(side="left", padx=(12, 8), pady=10)
        self.train_status = tk.Label(
            train_strip,
            text="Ready when weights are set (dataset splits: Data tab).",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.train_status.pack(side="left", fill="x", expand=True, pady=10, padx=(0, 12))

        # ── STEP 4 ───────────────────────────────────────────────
        sec6 = self._step_section(
            scrollable,
            4,
            "Ship it",
            "Smoke-test a checkpoint, export ONNX for deployment, or open the runs folder.",
        )
        sec6.pack(fill="x", padx=pad_x, pady=(8, 6))

        row6a = tk.Frame(sec6, bg=BG_PANEL)
        row6a.pack(fill="x", padx=18, pady=(4, 8))
        FlatButton(
            row6a,
            text="Validate",
            command=self._validate_model,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=118,
            height=34,
            font_size=9,
        ).pack(side="left", padx=(2, 8))
        FlatButton(
            row6a,
            text="Export ONNX",
            command=self._export_onnx,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=118,
            height=34,
            font_size=9,
        ).pack(side="left", padx=(0, 8))
        FlatButton(
            row6a,
            text="Open runs folder",
            command=self._open_runs,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=138,
            height=34,
            font_size=9,
        ).pack(side="left", padx=(0, 12))

        list_hdr = tk.Frame(sec6, bg=BG_PANEL)
        list_hdr.pack(fill="x", padx=18, pady=(0, 6))
        tk.Label(
            list_hdr,
            text="Checkpoints (best.pt)",
            bg=BG_PANEL,
            fg=FG,
            font=(FONT, 10, "bold"),
            anchor="w",
        ).pack(side="left")
        tk.Label(
            list_hdr,
            text="Detected under runs/",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 8),
            anchor="e",
        ).pack(side="right")

        self.models_list = tk.Listbox(
            sec6,
            bg=BG_PANEL,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            font=(MONO, 9),
            height=7,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            activestyle="none",
        )
        self.models_list.pack(fill="both", expand=True, padx=18, pady=(0, 28))
        self._refresh_models_list()

    # ── Section builder ────────────────────────────────────────

    def _step_section(self, parent, step_num: int, title: str, description: str) -> tk.Frame:
        """Card-style step with accent rail and hierarchy."""
        outer = tk.Frame(parent, bg=BG)
        card = tk.Frame(
            outer,
            bg=BG_CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.pack(fill="x")

        head = tk.Frame(card, bg=BG_CARD)
        head.pack(fill="x", padx=0, pady=(16, 8))
        rail = tk.Frame(head, bg=ACCENT, width=4)
        rail.pack(side="left", fill="y", padx=(16, 0))
        rail.pack_propagate(False)

        tit_col = tk.Frame(head, bg=BG_CARD)
        tit_col.pack(side="left", fill="x", expand=True, padx=(14, 16))
        tk.Label(
            tit_col,
            text=f"STEP {step_num}",
            bg=BG_CARD,
            fg=ACCENT,
            font=(FONT, 8, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            tit_col,
            text=title,
            bg=BG_CARD,
            fg=FG,
            font=(FONT, 13, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        tk.Label(
            card,
            text=description,
            bg=BG_CARD,
            fg=FG_DIM,
            font=(FONT, 10),
            anchor="w",
            wraplength=880,
            justify="left",
        ).pack(fill="x", padx=(34, 22), pady=(0, 12))

        return outer

    # ── Model Selection ──────────────────────────────────────────────

    def _browse_model(self):
        """Browse for custom model weights."""
        model_file = filedialog.askopenfilename(
            title="Choose YOLO model (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if model_file:
            self.model_type_var.set("Custom weights...")
            self.current_model_path = Path(model_file)
            self.model_path_lbl.configure(text=f"Using: {self.current_model_path.name}")
            self._refresh_models_list()

    def _get_model_source(self) -> str:
        """Get the model source path based on selection."""
        choice = self.model_type_var.get()
        if "New Model" in choice:
            return "yolov8s-obb.pt"
        elif "v1" in choice:
            return str(RUNS_DIR / "obb_v1/poi_obb_v1/weights/best.pt")
        else:
            return str(self.current_model_path) if self.current_model_path else "yolov8s-obb.pt"

    # ── Training Control ──────────────────────────────────────────────

    def _start_training(self):
        """Start YOLO training with current settings."""
        if self.training:
            messagebox.showwarning("Training Active", "Training is already running.")
            return

        # Validate inputs
        try:
            lr = float(self.lr_var.get())
            batch = int(self.batch_var.get())
            epochs = int(self.epochs_var.get())
            imgsz = int(self.imgsz_var.get())
            patience = int(self.patience_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for hyperparameters.")
            return

        model_src = self._get_model_source()
        if not Path(model_src).exists() and "yolov" not in model_src:
            messagebox.showerror("Model Not Found", f"Model not found: {model_src}")
            return

        # Confirm
        if not messagebox.askyesno("Start Training",
                                    f"Start training with:\n"
                                    f"  Model: {model_src}\n"
                                    f"  LR: {lr}, Batch: {batch}\n"
                                    f"  Epochs: {epochs}, Image Size: {imgsz}"):
            return

        self.training = True
        self.training_stop_event.clear()
        self.start_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER)
        self.stop_btn.configure_colors(bg=RED, bg_hover=RED_HOVER)
        self.train_status.configure(text="Starting training...")

        def _train_worker():
            try:
                model = YOLO(model_src)
                results = model.train(
                    data=str(OUTPUTS_DIR / "poi_obb_data.yaml"),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    lr0=lr,
                    lrf=0.01,
                    patience=patience,
                    device=0,
                    project=str(RUNS_DIR / "obb_v7"),
                    name="poi_obb_v7",
                    exist_ok=True,
                    resume=False,
                    amp=True,
                    close_mosaic=10,
                    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
                    degrees=15, translate=0.2, scale=0.9,
                    shear=5.0, fliplr=0.5, flipud=0.1,
                    mosaic=0.5, auto_augment="randaugment",
                    dropout=0.15, cls=0.5, box=7.5, angle=1.0,
                    warmup_epochs=3.0, warmup_bias_lr=0.1,
                )
                self._tab_ui_queue.put(("train_complete", True, "Training complete!"))
            except Exception as e:
                self._tab_ui_queue.put(("train_complete", False, f"Error: {e}"))

        self.training_thread = threading.Thread(target=_train_worker, daemon=True)
        self.training_thread.start()

    def _stop_training(self):
        """Stop training."""
        if not self.training:
            return
        self.training_stop_event.set()
        self.train_status.configure(text="Stopping training...")
        # Note: YOLO doesn't support graceful stop, but we can mark as stopped
        self._training_complete(False, "Training stopped by user")

    def _training_complete(self, success: bool, message: str):
        """Handle training completion."""
        self.training = False
        self.start_btn.configure_colors(bg=ACCENT, bg_hover=ACCENT_HOVER)
        self.stop_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER)
        self.train_status.configure(text=message)
        self.progress_var.set(0)
        self._refresh_models_list()
        if success:
            try:
                # Try to locate latest trained model
                candidates = list(RUNS_DIR.rglob("best.pt"))
                if candidates:
                    last_model = max(candidates, key=lambda p: p.stat().st_mtime)
                    if last_model.exists():
                        if messagebox.askyesno("Load Model", "Load this model on the Run tab now?"):
                            self.main_app.load_trained_model(str(last_model))
                else:
                    messagebox.showinfo("Training Complete", message)
            except Exception as e:
                messagebox.showinfo("Training Complete", f"Training done. Could not load model: {e}")
        else:
            messagebox.showwarning("Training Stopped", message)

    # ── Model Management ────────────────────────────────────────────

    def _validate_model(self):
        """Validate a model checkpoint."""
        model_file = filedialog.askopenfilename(
            title="Select Model to Validate",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if not model_file:
            return

        def _worker():
            try:
                model = YOLO(model_file)
                test_img = Path("data/images/test/images_1777558649281.jpg")
                if not test_img.exists():
                    test_img = list(Path("data/images/test").glob("*.jpg"))[0]
                results = model(str(test_img), conf=0.25, imgsz=960)
                r = results[0]
                if hasattr(r, 'obb') and r.obb is not None:
                    confs = r.obb.conf.cpu().numpy()
                    msg = f"✅ Model works! {len(confs)} detections, top conf={confs[0]:.4f}"
                else:
                    msg = "❌ Model broken - no detections with conf=0.25"
                self._tab_ui_queue.put(("train_status", msg))
            except Exception as e:
                self._tab_ui_queue.put(("train_status", f"❌ Validation error: {e}"))

        self.train_status.configure(text="Validating model...")
        threading.Thread(target=_worker, daemon=True).start()

    def _export_onnx(self):
        """Export model to ONNX format."""
        model_file = filedialog.askopenfilename(
            title="Select Model to Export",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if not model_file:
            return

        def _worker():
            try:
                model = YOLO(model_file)
                model.export(format="onnx")
                self._tab_ui_queue.put(("train_status", "✅ Model exported to ONNX format"))
            except Exception as e:
                self._tab_ui_queue.put(("train_status", f"❌ Export error: {e}"))

        self.train_status.configure(text="Exporting to ONNX...")
        threading.Thread(target=_worker, daemon=True).start()

    def _open_runs(self):
        """Open runs/ folder."""
        import os
        runs_path = RUNS_DIR / "obb_v7"
        if runs_path.exists():
            os.startfile(str(runs_path))
        else:
            messagebox.showinfo("Not Found", "No training runs yet. Train a model first.")

    def _refresh_models_list(self):
        """Refresh the list of available models."""
        self.models_list.delete(0, "end")
        try:
            for p in RUNS_DIR.rglob("best.pt"):
                if "obb" in str(p).lower():
                    self.models_list.insert("end", str(p))
        except Exception:
            pass

    def _on_close(self) -> None:
        """App shutdown: stop OBB training thread if still running."""
        if self._tab_ui_poll_id is not None:
            try:
                self.main_app.root.after_cancel(self._tab_ui_poll_id)
            except tk.TclError:
                pass
            self._tab_ui_poll_id = None
        self.training_stop_event.set()
        if self.training_thread is not None and self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)




class PoiStudio:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("POI Studio - Detection Platform")
        self.root.geometry("1500x920")
        self.root.minsize(1200, 700)
        self.root.configure(bg=BG)

        tab_bar = tk.Frame(root, bg=BG_PANEL, height=40)
        tab_bar.pack(fill=tk.X, side=tk.TOP)
        tab_bar.pack_propagate(False)

        self._tab_frames: list[tuple[tk.Frame, str]] = []
        self._tab_buttons: list[tk.Button] = []
        self._current_tab = 0

        def make_tab(name: str, index: int) -> tk.Frame:
            frame = tk.Frame(root, bg=BG)
            frame.pack(fill=tk.BOTH, expand=True)
            frame.pack_forget()
            self._tab_frames.append((frame, name))
            return frame

        viewer_tab = make_tab("Run", 0)
        data_tab = make_tab("Data", 1)
        train_tab = make_tab("Train", 2)

        def create_tab_btn(text: str, index: int) -> tk.Button:
            btn = tk.Button(tab_bar, text=text, bg=BG_CARD, fg=FG, font=(FONT, 10, "bold"),
                           relief=tk.FLAT, bd=0, padx=20, pady=8,
                           command=lambda i=index: self._switch_tab(i))
            btn.pack(side=tk.LEFT, padx=2, pady=4)
            return btn

        self._tab_buttons.append(create_tab_btn("Run", 0))
        self._tab_buttons.append(create_tab_btn("Data", 1))
        self._tab_buttons.append(create_tab_btn("Train", 2))

        self.notebook = type('obj', (object,), {'select': lambda i: None, 'index': lambda s: len(self._tab_frames)})()

        self.viewer = PoiDesktopViewer(viewer_tab, embedded=True, bind_shortcuts=False)
        self.data_tab_shell = DataManagementTab(data_tab, self)
        self.relabel = self.data_tab_shell.relabel
        self.trainer = TrainingTab(train_tab, self)
        self._switch_tab(0)
        self.viewer.set_capture_for_relabel_callback(self._open_capture_in_relabel)
        self.relabel.set_save_to_train_callback(self._save_relabel_to_train)
        self.relabel.set_ai_assist_callback(self._predict_for_relabel)

        self._bind_scoped_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.viewer.model is None:
            messagebox.showwarning(
                "Model missing",
                "Default model not found. Click 'Model' and select your best.pt file.",
            )

    def _switch_tab(self, index: int) -> None:
        for i, (frame, _) in enumerate(self._tab_frames):
            if i == index:
                frame.pack(fill=tk.BOTH, expand=True)
                self._tab_buttons[i].configure(bg=BG_PANEL, fg=ACCENT)
            else:
                frame.pack_forget()
                self._tab_buttons[i].configure(bg=BG_CARD, fg=FG)
        self._current_tab = index
        self._on_tab_changed()

    def _active_tab_name(self) -> str:
        return self._tab_frames[self._current_tab][1]

    def _bind_scoped_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda _: self.viewer._prev_image() if self._active_tab_name() == "Run" else self.relabel.prev_image())
        self.root.bind("<Right>", lambda _: self.viewer._next_image() if self._active_tab_name() == "Run" else self.relabel.next_image())
        self.root.bind("<space>", lambda _: self.viewer._toggle_autolabel() if self._active_tab_name() == "Run" else None)
        self.root.bind("<Control-s>", lambda _: self.relabel.save_all() if self._active_tab_name() == "Data" else None)
        self.root.bind("<Control-z>", lambda _: self.relabel.undo_point() if self._active_tab_name() == "Data" else None)
        self.root.bind("<Delete>", lambda _: self.relabel._delete_selected_poi() if self._active_tab_name() == "Data" else None)
        self.root.bind("<Escape>", lambda _: self.relabel._cancel_add() if self._active_tab_name() == "Data" else None)
        self.root.bind("<Return>", lambda _: self.relabel._finish_add_if_ready() if self._active_tab_name() == "Data" else None)
        self.root.bind("<a>", lambda _: self.relabel._start_add_poi() if self._active_tab_name() == "Data" else None)
        self.root.bind("<plus>", lambda _: self.relabel._zoom_in() if self._active_tab_name() == "Data" else None)
        self.root.bind("<equal>", lambda _: self.relabel._zoom_in() if self._active_tab_name() == "Data" else None)
        self.root.bind("<minus>", lambda _: self.relabel._zoom_out() if self._active_tab_name() == "Data" else None)
        self.root.bind("<Control-0>", lambda _: self.relabel._zoom_reset() if self._active_tab_name() == "Data" else None)

    def _on_tab_changed(self, _event=None) -> None:
        if self._active_tab_name() != "Run":
            self.viewer._stop_live()

    def _open_capture_in_relabel(self, image_path: Path, labels_root: Path) -> None:
        self._switch_tab(1)
        self.data_tab_shell.focus_label_tab()
        self.relabel.open_specific_image(image_path, labels_root=labels_root)

    def _predict_for_relabel(self, bgr_image: np.ndarray) -> list[list[tuple[int, int]]]:
        if not self.viewer.model:
            raise RuntimeError("YOLO model is not loaded on the Run tab.")
            
        result = self.viewer.model.predict(
            source=bgr_image, device=self.viewer.device, conf=0.50, imgsz=960, verbose=False
        )[0]
        
        points_list = []
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            for corners in obb.xyxyxyxy.cpu().numpy():
                quad = self.viewer._order_quad_points(corners.astype(np.float32))
                pts = [(int(x), int(y)) for x, y in quad]
                points_list.append(pts)
        return points_list

    def _save_relabel_to_train(self, image_path: Path, label_path: Path | None) -> str:
        master_images = Path("data_master/labeled_with_poi/images")
        master_labels = Path("data_master/labeled_with_poi/labels")
        master_images.mkdir(parents=True, exist_ok=True)
        master_labels.mkdir(parents=True, exist_ok=True)

        out_image = master_images / image_path.name
        out_label = master_labels / f"{image_path.stem}.txt"
        
        # Move the image so it leaves the unlabeled queue
        if image_path.exists():
            shutil.move(str(image_path), str(out_image))
            
        if label_path is not None and label_path.exists():
            shutil.move(str(label_path), str(out_label))
            return f"Moved to labeled_with_poi/{out_image.name}"
            
        # Ensure a negative has an empty label file in the master dir
        if not out_label.exists():
            out_label.touch()
        return f"Moved negative to labeled_with_poi/{out_image.name}"

    def load_trained_model(self, path: str) -> None:
        """Load a trained model on the Run tab safely."""
        try:
            p = Path(path)
            if not p.exists():
                self.viewer.status_var.set(f"Model not found: {path}")
                return
            self.viewer.model_path = p
            self.viewer._load_model()
            if self.viewer.current_image_path and self.viewer.model is not None:
                self.viewer._run_prediction(self.viewer.current_image_path)
        except Exception as e:
            self.viewer.status_var.set(f"Failed to load trained model: {e}")

    def _on_close(self) -> None:
        self.viewer._on_close()
        self.data_tab_shell._on_close()
        self.trainer._on_close()
        self.root.destroy()


# ===== MERGED FROM relabel_corners_tool.py =====


from pathlib import Path
from tkinter import filedialog, messagebox


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Theme ────────────────────────────────────────────────────────────────
BG        = "#0f0f0f"
BG_PANEL  = "#1a1a1a"
BG_CARD   = "#222222"
BG_INPUT  = "#2a2a2a"
BG_HOVER  = "#333333"
BG_ACTIVE = "#3a3a3a"
FG        = "#e0e0e0"
FG_DIM    = "#888888"
FG_SEL    = "#ffffff"
ACCENT    = "#00a86b"
ACCENT_HV = "#00c07e"
RED       = "#e04040"
RED_HV    = "#f05050"
YELLOW    = "#e0c040"
CYAN      = "#40c0e0"
ORANGE    = "#e0952a"
ORANGE_HV = "#f0a53a"
BORDER    = "#333333"
SEP       = "#2a2a2a"
FONT      = "Segoe UI"
MONO      = "Consolas"

# ── Unified color palette (same as desktop_poi_viewer) ───────────────────
# BGR values used by OpenCV drawing
POI_COLORS_BGR = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 200, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 0, 255),      # red
    (200, 200, 0),    # teal
    (128, 0, 128),    # purple
]

# Hex equivalents for tkinter canvas
def _bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

POI_COLORS_HEX = [_bgr_to_hex(c) for c in POI_COLORS_BGR]

DRAG_RADIUS = 8


# ── Styled widgets ───────────────────────────────────────────────────────

class FlatButton(tk.Canvas):
    def __init__(
        self, parent, *, text="", command=None, bg=ACCENT, bg_hover=ACCENT_HV,
        fg="#ffffff", width=90, height=32, font_size=9, bold=False, radius=6,
    ) -> None:
        self._bg = bg
        self._bg_hover = bg_hover
        self._fg = fg
        self._text = text
        self._command = command
        self._radius = radius
        weight = "bold" if bold else "normal"
        self._font = (FONT, font_size, weight)
        parent_bg = BG_PANEL
        if isinstance(parent, (tk.Frame, tk.Canvas)):
            try:
                parent_bg = parent.cget("bg")
            except Exception:
                pass
        super().__init__(parent, width=width, height=height,
                         bg=parent_bg, highlightthickness=0, bd=0)
        self._draw(self._bg)
        self.bind("<Enter>", lambda _: self._draw(self._bg_hover))
        self.bind("<Leave>", lambda _: self._draw(self._bg))
        self.bind("<ButtonPress-1>", lambda _: self._click())
        self.bind("<ButtonRelease-1>", lambda _: self._draw(self._bg_hover))

    def _draw(self, color: str) -> None:
        self.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        r = self._radius
        pts = [r,0,w-r,0,w,0,w,r, w,h-r,w,h,w-r,h, r,h,0,h,0,h-r,0,r,0,0]
        self.create_polygon(pts, smooth=True, fill=color, outline="")
        self.create_text(w//2, h//2, text=self._text, fill=self._fg, font=self._font)

    def _click(self) -> None:
        if self._command:
            self._command()

    def configure_text(self, text: str) -> None:
        self._text = text
        self._draw(self._bg)

    def configure_colors(self, *, bg=None, bg_hover=None, fg=None):
        if bg: self._bg = bg
        if bg_hover: self._bg_hover = bg_hover
        if fg: self._fg = fg
        self._draw(self._bg)


# ── Label parsing ────────────────────────────────────────────────────────

def _parse_label(text: str) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        values = [float(v) for v in parts[1:]]
        if len(values) < 4 or len(values) % 2 != 0:
            continue
        pts: list[tuple[float, float]] = []
        for i in range(0, len(values), 2):
            pts.append((values[i], values[i + 1]))
        polygons.append(pts)
    return polygons


def _poly_to_pixels(poly, w, h):
    return [(int(x * w), int(y * h)) for x, y in poly]


def _poly_to_str(poly, w, h):
    tokens = []
    for px, py in poly:
        tokens.append(f"{px / w:.6f}")
        tokens.append(f"{py / h:.6f}")
    return "0 " + " ".join(tokens)


# ── Data model for editable POIs ────────────────────────────────────────

class EditablePoi:
    """One POI polygon with pixel-coordinate points (relative to full-res image)."""
    def __init__(self, points: list[tuple[int, int]], color_idx: int = 0):
        self.points: list[tuple[int, int]] = list(points)
        self.color_idx = color_idx


# ── Main tool ────────────────────────────────────────────────────────────

class RelabelCornersTool:
    def __init__(self, root: tk.Misc, *, embedded: bool = False, bind_shortcuts: bool = True) -> None:
        self.root = root
        self.embedded = embedded
        self.window = root if isinstance(root, tk.Tk) else root.winfo_toplevel()
        if not self.embedded:
            self.window.title("Relabel POI Corners")
            self.window.geometry("1480x880")
            self.window.minsize(1100, 650)
            self.window.configure(bg=BG)

        self.images_root = Path("data").resolve()
        self.labels_root = Path("data/labels").resolve()

        self.image_paths: list[Path] = []
        self.index = 0
        self.current_image: np.ndarray | None = None
        self.current_tk: ImageTk.PhotoImage | None = None
        self.preview_tk: ImageTk.PhotoImage | None = None
        self.render_info = {"x": 0, "y": 0, "w": 1, "h": 1}

        # Editable POIs for current image (loaded from label file)
        self.pois: list[EditablePoi] = []
        self.selected_poi: int = -1
        self.dragging: tuple[int, int] | None = None

        # New-POI click state (stored as IMAGE coordinates)
        self.new_points_canvas: list[tuple[int, int]] = []

        # Zoom / pan state
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_start: tuple[int, int] | None = None
        self._pan_start_offset: tuple[float, float] | None = None
        self._ZOOM_MIN = 0.5
        self._ZOOM_MAX = 16.0
        self._ZOOM_STEP = 1.15
        self._last_double_click_time = 0.0

        # Dirty tracking
        self._dirty = False
        self._dirty_clear_id = None

        # POI list panel
        self.poi_list_frame: tk.Frame | None = None
        self.poi_list_inner: tk.Frame | None = None
        self.poi_list_canvas: tk.Canvas | None = None
        self.poi_buttons: list[tk.Frame] = []
        self._save_to_train_callback = None
        self._ai_assist_callback = None

        self._build_ui()
        if bind_shortcuts:
            self._bind_shortcuts()

    def set_save_to_train_callback(self, callback) -> None:
        self._save_to_train_callback = callback

    def set_ai_assist_callback(self, callback) -> None:
        self._ai_assist_callback = callback

    def _default_export_to_train(self, image_path: Path, label_path: Path | None) -> str:
        data_root = Path("data")
        train_images = data_root / "images" / "train"
        train_labels = data_root / "labels" / "train"
        train_images.mkdir(parents=True, exist_ok=True)
        train_labels.mkdir(parents=True, exist_ok=True)

        out_image = train_images / image_path.name
        out_label = train_labels / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)
        if label_path is not None and label_path.exists():
            shutil.copy2(label_path, out_label)
            return f"Exported to train/{out_image.name}"
        if out_label.exists():
            out_label.unlink()
        return f"Exported negative to train/{out_image.name}"

    # ─────────────────────────────────────────────────────────── UI ──

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg=BG_PANEL, height=54)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        row = tk.Frame(header, bg=BG_PANEL, padx=14, pady=8)
        row.pack(fill=tk.BOTH, expand=True)

        tk.Label(row, text="Relabel", bg=BG_PANEL, fg=ACCENT,
                 font=(FONT, 14, "bold")).pack(side=tk.LEFT)
        tk.Label(row, text="Tool", bg=BG_PANEL, fg=FG,
                 font=(FONT, 14, "bold")).pack(side=tk.LEFT, padx=(0, 16))
        self._sep(row)

        FlatButton(row, text="Open Images Dir", command=self.open_folder,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=110, height=32
                   ).pack(side=tk.LEFT, padx=(8, 3))
        FlatButton(row, text="Set Labels Dir", command=self._choose_labels_dir,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=104, height=32
                   ).pack(side=tk.LEFT, padx=3)
        self._sep(row)

        FlatButton(row, text="\u25C0 Prev", command=self.prev_image,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=56, height=32
                   ).pack(side=tk.LEFT, padx=(8, 2))
        FlatButton(row, text="Next \u25B6", command=self.next_image,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=56, height=32
                   ).pack(side=tk.LEFT, padx=2)
        self._sep(row)

        FlatButton(row, text="Undo", command=self.undo_point,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=50, height=32
                   ).pack(side=tk.LEFT, padx=(8, 3))
        FlatButton(row, text="Clear", command=self.clear_points,
                   bg=BG_CARD, bg_hover=BG_HOVER, width=50, height=32
                   ).pack(side=tk.LEFT, padx=3)
        self._sep(row)

        FlatButton(row, text="Add POI", command=self._start_add_poi,
                   bg=ACCENT, bg_hover=ACCENT_HV, width=72, height=32, bold=True
                   ).pack(side=tk.LEFT, padx=(8, 3))
        
        self._ai_assist_btn = FlatButton(row, text="AI Assist", command=self._run_ai_assist,
                                        bg=ORANGE, bg_hover=ORANGE_HV, width=80, height=32, bold=True)
        self._ai_assist_btn.pack(side=tk.LEFT, padx=3)

        FlatButton(row, text="Delete POI", command=self._delete_selected_poi,
                   bg=RED, bg_hover=RED_HV, width=86, height=32, bold=True
                   ).pack(side=tk.LEFT, padx=3)
        FlatButton(row, text="Delete File", command=self.delete_label_file,
                   bg=RED, bg_hover=RED_HV, width=86, height=32, bold=True
                   ).pack(side=tk.LEFT, padx=3)

        self._labels_dir_lbl = tk.Label(
            row, text=f"Labels: {self.labels_root}", bg=BG_PANEL,
            fg=FG_DIM, font=(MONO, 8), anchor="e")
        self._labels_dir_lbl.pack(side=tk.RIGHT, padx=(8, 0))

        # Status bar
        status_bar = tk.Frame(self.root, bg=BG_PANEL, height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        status_inner = tk.Frame(status_bar, bg=BG_PANEL, padx=14)
        status_inner.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(
            value="Open an images folder to begin. Click POIs to select/edit. Drag points to adjust."
        )
        tk.Label(status_inner, textvariable=self.status_var, bg=BG_PANEL,
                 fg=FG_DIM, font=(MONO, 9), anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)

        # Dirty indicator in status bar
        self._dirty_lbl = tk.Label(status_inner, text="", bg=BG_PANEL,
                                   fg=ORANGE, font=(MONO, 9, "bold"))
        self._dirty_lbl.pack(side=tk.RIGHT, padx=(8, 0), pady=4)

        # Main area: left canvas | middle POI list | right preview
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        main.grid_columnconfigure(0, weight=4)
        main.grid_columnconfigure(1, weight=0, minsize=180)
        main.grid_columnconfigure(2, weight=3)
        main.grid_rowconfigure(1, weight=1)

        # Section headers
        for col, (title, color) in enumerate([
            ("Image \u2014 click POI to select, drag points to adjust", CYAN),
            ("POIs", ACCENT),
            ("Preview", YELLOW),
        ]):
            hdr = tk.Frame(main, bg=BG)
            hdr.grid(row=0, column=col, sticky="ew", pady=(0, 4),
                     padx=(0 if col == 0 else 6, 0))
            tk.Frame(hdr, bg=color, width=3, height=14).pack(side=tk.LEFT, padx=(0, 8))
            tk.Label(hdr, text=title, bg=BG, fg=FG, font=(FONT, 10, "bold")).pack(side=tk.LEFT)

        # Left canvas
        card_left = tk.Frame(main, bg=BORDER, bd=0)
        card_left.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.canvas = tk.Canvas(card_left, bg=BG_CARD, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", lambda e: self._on_scroll_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._on_scroll_linux(e, -1))
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

        # Middle — POI list panel (scrollable)
        self.poi_list_frame = tk.Frame(main, bg=BG_PANEL, width=180, bd=0,
                                       highlightthickness=1, highlightbackground=BORDER)
        self.poi_list_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 6))
        self.poi_list_frame.grid_propagate(False)

        list_header = tk.Frame(self.poi_list_frame, bg=BG_PANEL)
        list_header.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(list_header, text="POI List", bg=BG_PANEL, fg=FG,
                 font=(FONT, 10, "bold")).pack(side=tk.LEFT)

        # Scrollable POI list using a canvas + inner frame
        list_outer = tk.Frame(self.poi_list_frame, bg=BG_PANEL)
        list_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self.poi_list_canvas = tk.Canvas(list_outer, bg=BG_PANEL, highlightthickness=0, bd=0)
        list_scrollbar = tk.Scrollbar(list_outer, orient=tk.VERTICAL,
                                      command=self.poi_list_canvas.yview)
        self.poi_list_canvas.configure(yscrollcommand=list_scrollbar.set)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.poi_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.poi_list_inner = tk.Frame(self.poi_list_canvas, bg=BG_PANEL)
        self.poi_list_canvas.create_window((0, 0), window=self.poi_list_inner, anchor="nw",
                                           tags="inner")
        self.poi_list_inner.bind("<Configure>",
            lambda _: self.poi_list_canvas.configure(
                scrollregion=self.poi_list_canvas.bbox("all")))
        self.poi_list_canvas.bind("<Configure>",
            lambda e: self.poi_list_canvas.itemconfigure("inner", width=e.width))

        # Mousewheel scrolling on POI list
        def _on_poi_list_mousewheel(event):
            self.poi_list_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.poi_list_canvas.bind("<MouseWheel>", _on_poi_list_mousewheel)
        self.poi_list_inner.bind("<MouseWheel>", _on_poi_list_mousewheel)

        # Right — preview canvas
        card_right = tk.Frame(main, bg=BORDER, bd=0)
        card_right.grid(row=1, column=2, sticky="nsew")
        self.preview_canvas = tk.Canvas(card_right, bg=BG_CARD, highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.preview_canvas.bind("<Configure>", lambda _: self.render())

    @staticmethod
    def _sep(parent):
        tk.Frame(parent, bg=SEP, width=1, height=22).pack(side=tk.LEFT, padx=6, pady=4)

    def _bind_shortcuts(self):
        self.window.bind("<Right>", lambda _: self.next_image())
        self.window.bind("<Left>", lambda _: self.prev_image())
        self.window.bind("<Control-s>", lambda _: self.save_all())
        self.window.bind("<Control-z>", lambda _: self.undo_point())
        self.window.bind("<Delete>", lambda _: self._delete_selected_poi())
        self.window.bind("<Escape>", lambda _: self._cancel_add())
        self.window.bind("<Return>", lambda _: self._finish_add_if_ready())
        self.window.bind("<a>", lambda _: self._start_add_poi())
        self.window.bind("<plus>", lambda _: self._zoom_in())
        self.window.bind("<equal>", lambda _: self._zoom_in())
        self.window.bind("<minus>", lambda _: self._zoom_out())
        self.window.bind("<Control-0>", lambda _: self._zoom_reset())

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered

    # ─────────────────────────────────────────────── Dirty tracking ──

    def _mark_dirty(self) -> None:
        self._auto_save()
        self._dirty_lbl.configure(text="\u2713 saved", fg=ACCENT)
        if hasattr(self, "_dirty_clear_id") and self._dirty_clear_id:
            self.window.after_cancel(self._dirty_clear_id)
        self._dirty_clear_id = self.window.after(2000, self._mark_clean)

    def _mark_clean(self) -> None:
        self._dirty = False
        self._dirty_clear_id = None
        self._dirty_lbl.configure(text="")

    def _choose_labels_dir(self):
        folder = filedialog.askdirectory(
            title="Select labels directory", initialdir=str(self.labels_root.parent))
        if not folder:
            return
        self.labels_root = Path(folder).resolve()
        self._labels_dir_lbl.configure(text=f"Labels: {self.labels_root}")
        self._reload_current()

    def open_folder(self):
        selected = filedialog.askdirectory(
            title="Select images folder", initialdir=str(self.images_root))
        if not selected:
            return
        base = Path(selected).resolve()
        images = [p for p in sorted(base.rglob("*"))
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not images:
            messagebox.showwarning("No images", "No supported images found.")
            return
        self.image_paths = images
        self.images_root = base
        self.index = 0
        self.new_points_canvas = []
        self._auto_detect_labels_dir()
        self._reload_current()

    def open_specific_image(self, image_path: Path, labels_root: Path | None = None) -> None:
        """Load a specific image into the editor and align label directory."""
        target = image_path.resolve()
        images_root = target.parent
        if target.parent.name == "images":
            images_root = target.parent
            if labels_root is None:
                labels_root = target.parent.parent / "labels"
        self.images_root = images_root

        images = [p for p in sorted(images_root.rglob("*"))
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not images:
            messagebox.showwarning("No images", f"No supported images found in {images_root}.")
            return
        self.image_paths = images
        try:
            self.index = self.image_paths.index(target)
        except ValueError:
            self.index = 0

        if labels_root is not None:
            self.labels_root = labels_root.resolve()
            self._labels_dir_lbl.configure(text=f"Labels: {self.labels_root}")
        else:
            self._auto_detect_labels_dir()

        self.new_points_canvas = []
        self._reload_current()

    def _auto_detect_labels_dir(self):
        base = self.images_root
        if base.name == "images" and (base.parent / "labels").is_dir():
            self.labels_root = (base.parent / "labels").resolve()
            self._labels_dir_lbl.configure(text=f"Labels: {self.labels_root}")
            return
        parent = base.parent
        for candidate in [parent / "labels", parent / "labels_poly"]:
            if candidate.is_dir():
                self.labels_root = candidate.resolve()
                self._labels_dir_lbl.configure(text=f"Labels: {self.labels_root}")
                return

    def _label_path_for(self, image_path: Path) -> Path:
        resolved = image_path.resolve()
        if resolved.parent.name == "images" and self.labels_root.name == "labels":
            return self.labels_root / f"{resolved.stem}.txt"
        if self.images_root in resolved.parents:
            rel = resolved.relative_to(self.images_root)
            out = self.labels_root / rel.parent / f"{resolved.stem}.txt"
        else:
            out = self.labels_root / f"{resolved.stem}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    # ─────────────────────────────────────────────── Navigation ──

    def prev_image(self):
        if not self.image_paths:
            return
        self.index = max(0, self.index - 1)
        self.new_points_canvas = []
        self._reload_current()

    def next_image(self):
        if not self.image_paths:
            return
        self.index = min(len(self.image_paths) - 1, self.index + 1)
        self.new_points_canvas = []
        self._reload_current()

    def _reload_current(self):
        self.pois = []
        self.selected_poi = -1
        self.dragging = None
        self._zoom_reset()
        self._mark_clean()
        self._load_image()
        self._load_pois_for_current()
        self._rebuild_poi_list()
        self.render()

    def _load_image(self):
        """Load current image from disk and cache it in self.current_image."""
        if not self.image_paths:
            self.current_image = None
            return
        image_path = self.image_paths[self.index]
        img = cv2.imread(str(image_path))
        if img is None:
            self.current_image = None
            self.status_var.set(f"Failed to read {image_path.name}")
            return
        self.current_image = img

    # ───────────────────────────────────────── POI list panel ──

    def _rebuild_poi_list(self):
        for w in self.poi_list_inner.winfo_children():
            w.destroy()
        self.poi_buttons = []

        if not self.pois:
            tk.Label(self.poi_list_inner, text="No POIs", bg=BG_PANEL, fg=FG_DIM,
                     font=(FONT, 9)).pack(anchor="w", padx=4, pady=4)
            return

        for i, poi in enumerate(self.pois):
            color_hex = POI_COLORS_HEX[i % len(POI_COLORS_HEX)]
            is_sel = (i == self.selected_poi)

            btn_frame = tk.Frame(
                self.poi_list_inner, bg=BG_ACTIVE if is_sel else BG_CARD,
                bd=0, highlightthickness=1,
                highlightbackground=color_hex if is_sel else BORDER,
                cursor="hand2",
            )
            btn_frame.pack(fill=tk.X, pady=2, padx=2)

            dot = tk.Canvas(btn_frame, width=12, height=12, bg=BG_ACTIVE if is_sel else BG_CARD,
                            highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(6, 4), pady=6)
            dot.create_oval(2, 2, 10, 10, fill=color_hex, outline="")

            n_pts = len(poi.points)
            sel_marker = " \u25C4" if is_sel else ""
            tk.Label(btn_frame, text=f"POI {i + 1} ({n_pts} pts){sel_marker}",
                     bg=BG_ACTIVE if is_sel else BG_CARD,
                     fg=FG_SEL if is_sel else FG,
                     font=(FONT, 9, "bold" if is_sel else "normal")
                     ).pack(side=tk.LEFT, padx=(0, 4), pady=6)

            idx = i
            for w in (btn_frame, dot):
                w.bind("<ButtonPress-1>", lambda e, ii=idx: self._select_poi(ii))
            for child in btn_frame.winfo_children():
                child.bind("<ButtonPress-1>", lambda e, ii=idx: self._select_poi(ii))

            self.poi_buttons.append(btn_frame)

        tk.Label(self.poi_list_inner, text="+ Click 'Add POI'\n  or press A",
                 bg=BG_PANEL, fg=FG_DIM, font=(FONT, 8), justify="left"
                 ).pack(anchor="w", padx=6, pady=(8, 4))

    def _select_poi(self, idx: int):
        self.selected_poi = idx
        self.new_points_canvas = []
        self._rebuild_poi_list()
        self.render()

    # ───────────────────────────────────────── Load / Save ──

    def _load_pois_for_current(self):
        if not self.image_paths or self.current_image is None:
            return
        image_path = self.image_paths[self.index]
        label_path = self._label_path_for(image_path)
        if not label_path.exists():
            return
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            return
        h, w = self.current_image.shape[:2]
        raw_polys = _parse_label(text)
        for i, poly in enumerate(raw_polys):
            pts_px = _poly_to_pixels(poly, w, h)
            self.pois.append(EditablePoi(pts_px, color_idx=i))

    def _auto_save(self):
        """Write current labels to disk (called automatically on every change)."""
        if not self.image_paths or self.current_image is None:
            return
        self._finish_add_if_ready()

        image_path = self.image_paths[self.index]
        h, w = self.current_image.shape[:2]
        lines: list[str] = []
        for poi in self.pois:
            if len(poi.points) != 4:
                continue
            lines.append(_poly_to_str(poi.points, w, h))
        label_path = self._label_path_for(image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            if label_path.exists():
                label_path.unlink()

    def save_all(self):
        """Manual save (kept for Ctrl+S backward compat)."""
        self._mark_dirty()
        if not self.image_paths or self.current_image is None:
            return
        image_path = self.image_paths[self.index]
        label_path = self._label_path_for(image_path)
        try:
            export_fn = self._save_to_train_callback or self._default_export_to_train
            msg = export_fn(image_path, label_path if label_path.exists() else None)
            if msg:
                self.status_var.set(f"{self.status_var.get()} | {msg}")
        except Exception as exc:
            self.status_var.set(f"{self.status_var.get()} | Train export failed: {exc}")

    def delete_label_file(self):
        if not self.image_paths:
            return
        image_path = self.image_paths[self.index]
        label_path = self._label_path_for(image_path)
        if not label_path.exists():
            self.status_var.set(f"No label file: {image_path.name}")
            return
        if not messagebox.askyesno(
            "Delete label file",
            f"Permanently delete the label file for {image_path.name}?\n\n{label_path}",
            icon=messagebox.WARNING,
        ):
            return
        label_path.unlink()
        self.pois = []
        self.selected_poi = -1
        self.new_points_canvas = []
        self._mark_dirty()
        self._rebuild_poi_list()
        self.render()
        self.status_var.set(f"Deleted label file for {image_path.name}")
        
    def _run_ai_assist(self):
        if not self.image_paths or self.current_image is None:
            self.status_var.set("AI Assist: No image loaded.")
            return
        if not self._ai_assist_callback:
            self.status_var.set("AI Assist: Not connected to model.")
            return
            
        self.status_var.set("AI Assist: Predicting...")
        self.root.update_idletasks()
        try:
            points_list = self._ai_assist_callback(self.current_image)
            if not points_list:
                self.status_var.set("AI Assist: No POIs found.")
                return
            
            self.pois = []
            for i, pts in enumerate(points_list):
                self.pois.append(EditablePoi(pts, color_idx=i))
            
            self.selected_poi = 0
            self._rebuild_poi_list()
            self._redraw()
            self._mark_dirty()
            self.status_var.set(f"AI Assist: Auto-populated {len(points_list)} POIs.")
        except Exception as e:
            self.status_var.set(f"AI Assist Error: {e}")

    # ──────────────────────────────────── Add / Delete POI ──

    def _start_add_poi(self):
        self.new_points_canvas = []
        self.selected_poi = -1
        self._rebuild_poi_list()
        self.status_var.set("Click 4 corners for new POI (TL, TR, BR, BL). Enter to confirm, Esc to cancel.")
        self.render()

    def _cancel_add(self):
        self.new_points_canvas = []
        self.status_var.set("Cancelled.")
        self.render()

    def _finish_add_if_ready(self):
        if len(self.new_points_canvas) == 4:
            color_idx = len(self.pois) % len(POI_COLORS_HEX)
            self.pois.append(EditablePoi(list(self.new_points_canvas), color_idx))
            self.selected_poi = len(self.pois) - 1
            self.new_points_canvas = []
            self._mark_dirty()
            self._rebuild_poi_list()
            self.status_var.set(f"Added POI {len(self.pois)}")
            self.render()

    def _delete_selected_poi(self):
        if self.selected_poi < 0 or self.selected_poi >= len(self.pois):
            self.status_var.set("Select a POI first (click it on the list or image).")
            return
        self.pois.pop(self.selected_poi)
        for i, poi in enumerate(self.pois):
            poi.color_idx = i
        self.selected_poi = min(self.selected_poi, len(self.pois) - 1)
        self._mark_dirty()
        self._rebuild_poi_list()
        self.render()
        self.status_var.set(f"Deleted. {len(self.pois)} POI(s) remaining.")

    # ────────────────────────────────────── Zoom / Pan ──

    def _zoom_reset(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _zoom_in(self):
        self._apply_zoom_center(self._ZOOM_STEP)

    def _zoom_out(self):
        self._apply_zoom_center(1.0 / self._ZOOM_STEP)

    def _apply_zoom_center(self, factor: float):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self._zoom_at(cw / 2, ch / 2, factor)

    def _on_scroll(self, event):
        if event.delta > 0:
            factor = self._ZOOM_STEP
        else:
            factor = 1.0 / self._ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)

    def _on_scroll_linux(self, event, direction: int):
        factor = self._ZOOM_STEP if direction > 0 else 1.0 / self._ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, cx: float, cy: float, factor: float):
        old_zoom = self._zoom
        new_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, self._zoom * factor))
        if new_zoom == old_zoom:
            return
        ratio = new_zoom / old_zoom
        base_x = (cx - self._pan_x) / old_zoom
        base_y = (cy - self._pan_y) / old_zoom
        self._pan_x = cx - base_x * new_zoom
        self._pan_y = cy - base_y * new_zoom
        self._zoom = new_zoom
        self.render()

    def _on_double_click(self, event):
        self._last_double_click_time = time.monotonic()
        self._zoom_reset()
        self.new_points_canvas = []
        self.render()

    def _on_pan_start(self, event):
        self._panning = True
        self._pan_start = (event.x, event.y)
        self._pan_start_offset = (self._pan_x, self._pan_y)
        self.canvas.configure(cursor="fleur")

    def _on_pan_move(self, event):
        if not self._panning or self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_x = self._pan_start_offset[0] + dx
        self._pan_y = self._pan_start_offset[1] + dy
        self.render()

    def _on_pan_end(self, event):
        self._panning = False
        self._pan_start = None
        self.canvas.configure(cursor="crosshair")

    # ────────────────────────────────────── Canvas interaction ──

    def _on_press(self, event):
        if self.current_image is None:
            return

        # Check if clicking near an existing point (drag start)
        for poi_idx, poi in enumerate(self.pois):
            for pt_idx, (px, py) in enumerate(poi.points):
                cx, cy = self._image_to_canvas(px, py)
                if abs(event.x - cx) <= DRAG_RADIUS and abs(event.y - cy) <= DRAG_RADIUS:
                    self.dragging = (poi_idx, pt_idx)
                    self.selected_poi = poi_idx
                    self._mark_dirty()
                    self._rebuild_poi_list()
                    return

        # Check if clicking inside a polygon (select it)
        ix, iy = self._canvas_to_image(event.x, event.y)
        for poi_idx, poi in enumerate(self.pois):
            if len(poi.points) >= 3:
                pts_np = np.array(poi.points, dtype=np.int32)
                if cv2.pointPolygonTest(pts_np, (ix, iy), False) >= 0:
                    self.selected_poi = poi_idx
                    self.new_points_canvas = []
                    self._rebuild_poi_list()
                    self.render()
                    return

        # Otherwise, add a new point for new POI creation
        ix_clamped = max(0, min(self.current_image.shape[1] - 1, ix))
        iy_clamped = max(0, min(self.current_image.shape[0] - 1, iy))
        self.new_points_canvas.append((ix_clamped, iy_clamped))
        self._mark_dirty()
        if len(self.new_points_canvas) == 4:
            self._finish_add_if_ready()
        else:
            self.status_var.set(
                f"New POI: {len(self.new_points_canvas)}/4 corners. "
                "Enter to confirm, Esc to cancel.")
        self.render()

    def _on_drag(self, event):
        if self.dragging is None or self.current_image is None:
            return
        poi_idx, pt_idx = self.dragging
        ix, iy = self._canvas_to_image(event.x, event.y)
        h, w = self.current_image.shape[:2]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        self.pois[poi_idx].points[pt_idx] = (ix, iy)
        self.render()

    def _on_release(self, event):
        if self.dragging is not None:
            self.dragging = None
            self._mark_dirty()
            self.status_var.set("Point moved.")

    def _on_right_click(self, event):
        """Right-click a point to delete it from its POI."""
        if self.current_image is None:
            return
        for poi_idx, poi in enumerate(self.pois):
            for pt_idx, (px, py) in enumerate(poi.points):
                cx, cy = self._image_to_canvas(px, py)
                if abs(event.x - cx) <= DRAG_RADIUS and abs(event.y - cy) <= DRAG_RADIUS:
                    if len(poi.points) <= 3:
                        self.status_var.set(
                            f"Cannot delete point — POI {poi_idx + 1} needs at least 3 points. "
                            "Use 'Delete POI' to remove entirely.")
                        return
                    else:
                        poi.points.pop(pt_idx)
                    self._mark_dirty()
                    self._rebuild_poi_list()
                    self.render()
                    self.status_var.set(
                        f"Removed point from POI {poi_idx + 1}.")
                    return

    # ─────────────────────────────────────────────── Undo ──

    def undo_point(self):
        if self.new_points_canvas:
            self.new_points_canvas.pop()
            self.render()

    def clear_points(self):
        self.new_points_canvas = []
        self.render()

    # ─────────────────────────────────────── Canvas helpers ──

    def _canvas_to_image(self, cx: int, cy: int) -> tuple[int, int]:
        if self.current_image is None:
            return 0, 0
        ih, iw = self.current_image.shape[:2]
        rx = self.render_info["x"]
        ry = self.render_info["y"]
        rw = self.render_info["w"]
        rh = self.render_info["h"]
        nx = (cx - rx) / float(max(1, rw))
        ny = (cy - ry) / float(max(1, rh))
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        return int(nx * (iw - 1)), int(ny * (ih - 1))

    def _image_to_canvas(self, ix: int, iy: int) -> tuple[float, float]:
        if self.current_image is None:
            return 0.0, 0.0
        ih, iw = self.current_image.shape[:2]
        rx = self.render_info["x"]
        ry = self.render_info["y"]
        rw = self.render_info["w"]
        rh = self.render_info["h"]
        cx = rx + (ix / max(1, iw - 1)) * rw
        cy = ry + (iy / max(1, ih - 1)) * rh
        return cx, cy

    @staticmethod
    def _fit_image(bgr, canvas):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        w, h = image.size
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        fitted = image.resize((nw, nh), Image.Resampling.BILINEAR)
        return fitted, (cw - nw) // 2, (ch - nh) // 2, nw, nh

    def _fit_image_zoomed(self, bgr, canvas):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        w, h = image.size
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())

        base_scale = min(cw / w, ch / h)
        eff_scale = base_scale * self._zoom
        nw = max(1, int(w * eff_scale))
        nh = max(1, int(h * eff_scale))

        fitted = image.resize((nw, nh), Image.Resampling.BILINEAR)

        base_x = (cw - int(w * base_scale)) // 2
        base_y = (ch - int(h * base_scale)) // 2

        zoom_cx = cw / 2
        zoom_cy = ch / 2
        final_x = zoom_cx - (zoom_cx - base_x) * self._zoom + self._pan_x
        final_y = zoom_cy - (zoom_cy - base_y) * self._zoom + self._pan_y

        return fitted, final_x, final_y, nw, nh

    # ─────────────────────────────────────────── Rendering ──

    def render(self):
        self.canvas.delete("all")
        if not self.image_paths or self.current_image is None:
            if not self.image_paths:
                self.status_var.set("Open an images folder to begin.")
            return

        # Use cached self.current_image — no disk read
        img = self.current_image

        fitted, x, y, nw, nh = self._fit_image_zoomed(img, self.canvas)
        self.render_info = {"x": x, "y": y, "w": nw, "h": nh}
        self.current_tk = ImageTk.PhotoImage(fitted)
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.current_tk)

        for i, poi in enumerate(self.pois):
            color = POI_COLORS_HEX[i % len(POI_COLORS_HEX)]
            is_sel = (i == self.selected_poi)
            self._draw_poi_on_canvas(poi, color, is_sel)

        self._draw_new_points()

        zoom_pct = int(self._zoom * 100)
        if zoom_pct != 100:
            cw = max(1, self.canvas.winfo_width())
            zoom_text = f"{zoom_pct}%"
            self.canvas.create_text(cw - 10, 10, anchor=tk.NE, text=zoom_text,
                                    fill=ACCENT, font=(MONO, 11, "bold"))
            self.canvas.create_text(cw - 10, 28, anchor=tk.NE,
                                    text="Double-click to reset",
                                    fill=FG_DIM, font=(FONT, 8))

        image_path = self.image_paths[self.index]
        n_pois = len(self.pois)
        sel_txt = f" | Selected: POI {self.selected_poi + 1}" if self.selected_poi >= 0 else ""
        add_txt = f" | Adding: {len(self.new_points_canvas)}/4" if self.new_points_canvas else ""
        zoom_txt = f" | Zoom: {zoom_pct}%" if zoom_pct != 100 else ""
        self.status_var.set(
            f"{self.index + 1}/{len(self.image_paths)} | {image_path.name} | "
            f"{n_pois} POI(s){sel_txt}{add_txt}{zoom_txt} | "
            "Scroll=zoom  Mid-click=pan  Double-click=reset"
        )

        self._render_preview()

    def _draw_poi_on_canvas(self, poi: EditablePoi, color: str, selected: bool):
        if len(poi.points) < 2:
            return
        canvas_pts = [self._image_to_canvas(px, py) for px, py in poi.points]
        flat = [c for pt in canvas_pts for c in pt]

        if len(poi.points) >= 3:
            if selected:
                self.canvas.create_polygon(flat, fill=color, stipple="gray25", outline="")
            else:
                self.canvas.create_polygon(flat, fill=color, stipple="gray12", outline="")

        width = 3 if selected else 2
        self.canvas.create_polygon(flat, fill="", outline=color, width=width)

        for pt_idx, (px, py) in enumerate(canvas_pts):
            r = 5 if selected else 3
            self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                    fill=color, outline="#000000" if selected else "")
            if selected:
                self.canvas.create_text(px + 8, py - 8, text=str(pt_idx + 1),
                                        fill=color, anchor=tk.NW,
                                        font=(MONO, 8, "bold"))

        cx = sum(p[0] for p in canvas_pts) / len(canvas_pts)
        cy = sum(p[1] for p in canvas_pts) / len(canvas_pts)
        idx = poi.color_idx + 1
        tag = f"POI {idx}"
        self.canvas.create_text(cx, cy - 12, text=tag, fill=color,
                                font=(FONT, 9, "bold"), anchor=tk.S)

    def _draw_new_points(self):
        color = "#00ff66"
        for idx, (ix, iy) in enumerate(self.new_points_canvas):
            px, py = self._image_to_canvas(ix, iy)
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5,
                                    fill=color, outline="#000000", width=1)
            self.canvas.create_text(px + 10, py - 10, text=str(idx + 1),
                                    fill=color, anchor=tk.NW, font=(MONO, 9, "bold"))
        for i in range(len(self.new_points_canvas) - 1):
            x0, y0 = self._image_to_canvas(*self.new_points_canvas[i])
            x1, y1 = self._image_to_canvas(*self.new_points_canvas[i + 1])
            self.canvas.create_line(x0, y0, x1, y1, fill=color, width=2, dash=(4, 4))

    def _render_preview(self):
        self.preview_canvas.delete("all")
        if self.current_image is None:
            return

        preview = self.current_image.copy()
        h, w = preview.shape[:2]

        for i, poi in enumerate(self.pois):
            if len(poi.points) < 3:
                continue
            bgr = POI_COLORS_BGR[i % len(POI_COLORS_BGR)]
            pts_np = np.array(poi.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(preview, [pts_np], bgr)
            cv2.polylines(preview, [pts_np], True, bgr, 2, cv2.LINE_AA)
            for j, (px, py) in enumerate(poi.points):
                cv2.circle(preview, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(preview, str(j + 1), (px + 6, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in poi.points]))
            cy = int(np.mean([p[1] for p in poi.points]))
            cv2.putText(preview, f"POI {i + 1}", (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

        fitted, fx, fy, fw, fh = self._fit_image(preview, self.preview_canvas)
        self.preview_tk = ImageTk.PhotoImage(fitted)
        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())
        ox = (cw - fitted.width) // 2
        oy = (ch - fitted.height) // 2
        self.preview_canvas.create_image(ox, oy, anchor=tk.NW, image=self.preview_tk)

        header = f"{len(self.pois)} POI(s)" if self.pois else "No labels"
        self.preview_canvas.create_text(14, 14, anchor=tk.NW, fill=FG_DIM,
                                        text=header, font=(FONT, 9, "bold"))


# ===== MERGED FROM extract_assets.py =====

"""Extract POI assets from labeled images to create transparent PNGs.

Saves each POI as a cropped BGRA image with an eroded alpha mask, plus a
sidecar .txt with 4 corner coordinates (pixel-space, relative to crop).
"""

import argparse
from pathlib import Path


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
    output_dir: Path = Path("data_master/synthetic_assets"),
) -> dict:
    """Extract POI assets from labeled images.

    images_dirs: list of image folders (or a single folder).
    labels_dirs: matching list of label folders.  If None, each image folder
                 is assumed to have a sibling ``labels/`` or ``../labels/``.
    output_dir:  where to write cropped PNG assets.

    Returns dict with counts.
    """
    if images_dirs is None:
        images_dirs = [Path("data_master/labeled_with_poi/images")]
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

import argparse
import math
import random
from pathlib import Path


# Configuration
BG_DIR = Path("data_master/without_poi_negatives/images")
ASSETS_DIR = Path("data_master/synthetic_assets")
OUT_IMG_DIR = Path("data_master/synthetic_output/images")
OUT_LBL_DIR = Path("data_master/synthetic_output/labels")

TARGET_SIZE = 960
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Probability weights for number of POIs per image  (1, 2, 3)
MULTI_POI_WEIGHTS = [0.60, 0.30, 0.10]


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

    # --- Placement (try up to 8 times to find a non-overlapping spot) ---
    px, py = 0, 0
    placed = False
    for _ in range(8):
        max_x = max(0, TARGET_SIZE - aw)
        max_y = max(0, TARGET_SIZE - ah)
        px = random.randint(0, max_x)
        py = random.randint(0, max_y)

        candidate = (px, py, px + aw, py + ah)
        if _box_overlap_ok(candidate, occupied_boxes, max_overlap=0.25):
            placed = True
            break

    if not placed:
        # Just place it wherever it fits; overlap is acceptable
        max_x = max(0, TARGET_SIZE - aw)
        max_y = max(0, TARGET_SIZE - ah)
        px = random.randint(0, max_x)
        py = random.randint(0, max_y)

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


def _box_overlap_ok(
    box: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    max_overlap: float,
) -> bool:
    """Return True if *box* doesn't overlap any box in *others* by more than *max_overlap*."""
    bx1, by1, bx2, by2 = box
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    for ox1, oy1, ox2, oy2 in others:
        ix1 = max(bx1, ox1)
        iy1 = max(by1, oy1)
        ix2 = min(bx2, ox2)
        iy2 = min(by2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / ba > max_overlap:
                return False
    return True


def _find_start_index() -> int:
    """Find the next available index by scanning existing output files."""
    if not OUT_IMG_DIR.exists():
        return 0
    max_idx = -1
    for f in OUT_IMG_DIR.iterdir():
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
) -> None:
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    bg_files = _get_files(BG_DIR)
    asset_files = _get_files(ASSETS_DIR)

    if not bg_files:
        print(f"Error: No backgrounds found in {BG_DIR}")
        return
    if not asset_files:
        print(f"Error: No POI assets found in {ASSETS_DIR}")
        return

    start_idx = _find_start_index()
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
            asset_path = random.choice(asset_files)
            result = _place_one_poi(bg, asset_path, occupied)
            if result is None:
                continue
            _, label_line = result
            label_lines.append(label_line)

        if not label_lines:
            continue  # skip images where nothing could be placed

        # Save image
        out_name = f"synth_{idx:05d}"
        out_img_path = OUT_IMG_DIR / f"{out_name}.jpg"
        cv2.imwrite(str(out_img_path), bg)
        if on_saved is not None:
            on_saved(out_img_path)

        # Save label (one line per POI)
        label_text = "\n".join(label_lines) + "\n"
        (OUT_LBL_DIR / f"{out_name}.txt").write_text(label_text)

        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print(f"Generated {i + 1}/{num_images} images (index {start_idx}–{idx})")

    print("\nGeneration Complete!")
    print(f"Output saved to: {OUT_IMG_DIR.resolve()}")


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


import argparse
import random
from pathlib import Path

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


def main() -> None:
    try:
        root = tk.Tk()
        app = PoiStudio(root)
        root.mainloop()
    except Exception:
        # Lightweight crash log to help diagnose in release
        import traceback
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "app.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("App crashed. See logs/app.log for details.")


if __name__ == "__main__":
    main()
