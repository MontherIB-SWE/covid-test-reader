"""Desktop inference viewer (Run tab)."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab, ImageTk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO

from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    AUTOLABEL_MIN_INTERVAL,
    BG,
    BG_CARD,
    BG_HOVER,
    BG_INPUT,
    BG_PANEL,
    BORDER,
    CAM_DISCONNECT_THRESHOLD,
    DATA_AUTOLABEL_DIR,
    FG,
    FG_DIM,
    FONT,
    LIVE_TARGET_MS,
    MONO,
    ORANGE,
    ORANGE_HOVER,
    POI_COLORS,
    POI_COLORS_HEX,
    RED,
    RED_HOVER,
    RUNS_DIR,
    SEP,
    SUPPORTED_EXTENSIONS,
    _select_device,
    _set_capture_exclusion,
    resolve_model_path,
)
from studio.widgets import FlatButton, FlatLabel, FlatOptionMenu

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
        resolved = resolve_model_path()
        self.model_path = resolved if resolved is not None else Path("best.pt")
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
        self.autolabel_dir = DATA_AUTOLABEL_DIR
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
            color_hex = POI_COLORS_HEX[i % len(POI_COLORS_HEX)]
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

