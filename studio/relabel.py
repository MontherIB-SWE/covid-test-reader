"""Corner relabeling tool (Label sub-tab / standalone)."""
from __future__ import annotations

import shutil
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    BG_ACTIVE,
    BG_CARD,
    BG_HOVER,
    BG_PANEL,
    BORDER,
    CYAN,
    DATA_DIR,
    DATA_LABELED_DIR,
    DRAG_RADIUS,
    FG,
    FG_DIM,
    FG_SEL,
    FONT,
    MONO,
    ORANGE,
    ORANGE_HOVER,
    POI_COLORS_BGR,
    POI_COLORS_HEX,
    RED,
    RED_HOVER,
    SEP,
    SUPPORTED_EXTENSIONS,
    YELLOW,
)
from studio.widgets import FlatButton

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

        self.images_root = DATA_DIR.resolve()
        self.labels_root = (DATA_DIR / "labels").resolve()

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
        master_images = DATA_LABELED_DIR / "images"
        master_labels = DATA_LABELED_DIR / "labels"
        master_images.mkdir(parents=True, exist_ok=True)
        master_labels.mkdir(parents=True, exist_ok=True)

        out_image = master_images / image_path.name
        out_label = master_labels / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)
        if label_path is not None and label_path.exists():
            shutil.copy2(label_path, out_label)
            return f"Copied to labeled/{out_image.name}"
        if not out_label.exists():
            out_label.touch()
        return f"Exported negative to labeled/{out_image.name}"

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
                   bg=ACCENT, bg_hover=ACCENT_HOVER, width=72, height=32, bold=True
                   ).pack(side=tk.LEFT, padx=(8, 3))
        
        self._ai_assist_btn = FlatButton(row, text="AI Assist", command=self._run_ai_assist,
                                        bg=ORANGE, bg_hover=ORANGE_HOVER, width=80, height=32, bold=True)
        self._ai_assist_btn.pack(side=tk.LEFT, padx=3)

        FlatButton(row, text="Delete POI", command=self._delete_selected_poi,
                   bg=RED, bg_hover=RED_HOVER, width=86, height=32, bold=True
                   ).pack(side=tk.LEFT, padx=3)
        FlatButton(row, text="Delete File", command=self.delete_label_file,
                   bg=RED, bg_hover=RED_HOVER, width=86, height=32, bold=True
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
        parent = self.window
        initial = str(self.labels_root.parent if self.labels_root.is_dir() else Path.home())
        folder = filedialog.askdirectory(
            title="Select labels directory",
            initialdir=initial,
            parent=parent,
            mustexist=True,
        )
        if not folder:
            return
        self.labels_root = Path(folder).resolve()
        self._labels_dir_lbl.configure(text=f"Labels: {self.labels_root}")
        self._reload_current()

    def open_folder(self):
        parent = self.window
        initial = str(self.images_root if self.images_root.is_dir() else Path.home())
        selected = filedialog.askdirectory(
            title="Select images folder",
            initialdir=initial,
            parent=parent,
            mustexist=True,
        )
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
            self.render()
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


