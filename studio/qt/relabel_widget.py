"""Label tab: corner relabeling (PySide6)."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from studio.qt.widgets import (
    make_button,
    make_menu_button,
    normalize_toolbar_widget,
    toolbar_panel,
    toolbar_row,
    toolbar_separator,
)
from studio.config import (
    ACCENT,
    BG_CARD,
    BG_PANEL,
    BORDER,
    CYAN,
    DATA_DIR,
    DATA_LABELED_DIR,
    DRAG_RADIUS,
    FG_DIM,
    FONT,
    MONO,
    ORANGE,
    POI_COLORS_BGR,
    POI_COLORS_HEX,
    SUPPORTED_EXTENSIONS,
    YELLOW,
)


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


class EditablePoi:
    def __init__(self, points: list[tuple[int, int]], color_idx: int = 0):
        self.points: list[tuple[int, int]] = list(points)
        self.color_idx = color_idx


class _EditorCanvas(QFrame):
    def __init__(self, owner: "RelabelWidget") -> None:
        super().__init__()
        self._owner = owner
        self.setMinimumSize(400, 320)
        self.setStyleSheet(f"background:{BG_CARD}; border:1px solid {BORDER};")
        self.setMouseTracking(True)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(BG_CARD))
        o = self._owner
        if o.current_image is None or not o.image_paths:
            return
        img = o.current_image
        fitted, x, y, nw, nh = o._fit_image_zoomed_widget(img, w, h)
        rgb = np.array(fitted.convert("RGB"))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        painter.drawImage(int(x), int(y), qimg)
        for i, poi in enumerate(o.pois):
            color = QColor(POI_COLORS_HEX[i % len(POI_COLORS_HEX)])
            o._draw_poi_painter(painter, poi, color, i == o.selected_poi)
        o._draw_new_points_painter(painter)
        zoom_pct = int(o._zoom * 100)
        if zoom_pct != 100:
            painter.setPen(QColor(ACCENT))
            painter.setFont(QFont(MONO, 11, QFont.Weight.Bold))
            painter.drawText(w - 10, 20, Qt.AlignmentFlag.AlignRight, f"{zoom_pct}%")
            painter.setPen(QColor(FG_DIM))
            painter.setFont(QFont(FONT, 8))
            painter.drawText(w - 10, 38, Qt.AlignmentFlag.AlignRight, "Double-click to reset")

    def mousePressEvent(self, e) -> None:
        self._owner._canvas_mouse_press(e)

    def mouseMoveEvent(self, e) -> None:
        self._owner._canvas_mouse_move(e)

    def mouseReleaseEvent(self, e) -> None:
        self._owner._canvas_mouse_release(e)

    def wheelEvent(self, e) -> None:
        self._owner._canvas_wheel(e)

    def mouseDoubleClickEvent(self, e) -> None:
        self._owner._canvas_double_click(e)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._owner.render()


class RelabelWidget(QWidget):
    status_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.images_root = DATA_DIR.resolve()
        self.labels_root = (DATA_DIR / "labels").resolve()
        self.image_paths: list[Path] = []
        self.index = 0
        self.current_image: np.ndarray | None = None
        self.pois: list[EditablePoi] = []
        self.selected_poi = -1
        self.dragging: tuple[int, int] | None = None
        self.new_points_canvas: list[tuple[int, int]] = []
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_start: tuple[float, float] | None = None
        self._pan_start_offset: tuple[float, float] = (0.0, 0.0)
        self._ZOOM_MIN = 0.5
        self._ZOOM_MAX = 16.0
        self._ZOOM_STEP = 1.15
        self.render_info: dict = {"x": 0, "y": 0, "w": 1, "h": 1}
        self._save_to_train_callback = None
        self._ai_assist_callback = None
        self._dirty_timer: QTimer | None = None
        self._build_ui()
        self._set_status("Open an images folder to begin.")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        title = QLabel(
            "<span style='color:#00a86b;font-weight:bold;font-size:12pt'>Relabel</span> "
            "<span style='font-weight:600;font-size:12pt'>Tool</span>"
        )
        normalize_toolbar_widget(title)
        row1 = toolbar_row(
            title,
            toolbar_separator(),
            make_menu_button("File ▾", [
                ("Open images folder…", self.open_folder),
                ("Set labels folder…", self._choose_labels_dir),
                ("-", None),
                ("Delete current label file", self.delete_label_file)
            ], tooltip="File operations"),
            make_menu_button("Edit ▾", [
                ("Undo last point (Ctrl+Z)", self.undo_point),
                ("Clear current corners", self.clear_points)
            ], tooltip="Edit operations"),
            toolbar_separator(),
            make_button("◀", self.prev_image, style="compact", tooltip="Previous image (←)"),
            make_button("▶", self.next_image, style="compact", tooltip="Next image (→)"),
            toolbar_separator(),
            make_button("+ Add POI", self._start_add_poi, style="primary",
                        tooltip="Click 4 corners on image (A)"),
            make_button("AI assist", self._run_ai_assist, style="primary",
                        tooltip="Fill POIs from Run tab model"),
            make_button("Delete POI", self._delete_selected_poi, style="danger",
                        tooltip="Remove selected polygon (Del)"),
        )
        labels_row = QWidget()
        lr = QHBoxLayout(labels_row)
        lr.setContentsMargins(0, 0, 0, 0)
        self._labels_dir_lbl = QLabel(f"Labels: {self.labels_root}")
        self._labels_dir_lbl.setStyleSheet(f"color:{FG_DIM}; font-family:{MONO}; font-size:9pt;")
        self._labels_dir_lbl.setWordWrap(True)
        lr.addWidget(self._labels_dir_lbl, 1)
        root.addWidget(toolbar_panel(row1, labels_row))

        mid = QHBoxLayout()
        left_col = QVBoxLayout()
        hdr1 = QHBoxLayout()
        hdr1.addWidget(self._bar(CYAN))
        hdr1.addWidget(QLabel("<b>Image — click POI to select, drag points</b>"))
        hdr1.addStretch(1)
        left_col.addLayout(hdr1)
        self._canvas = _EditorCanvas(self)
        left_col.addWidget(self._canvas, 1)
        list_col = QVBoxLayout()
        list_col.addLayout(self._hdr_row("POIs", ACCENT))
        self._poi_list = QListWidget()
        self._poi_list.setMaximumWidth(220)
        self._poi_list.setStyleSheet(f"background:{BG_PANEL};")
        self._poi_list.itemClicked.connect(self._on_poi_list_click)
        list_col.addWidget(self._poi_list, 1)
        prev_col = QVBoxLayout()
        prev_col.addLayout(self._hdr_row("Preview", YELLOW))
        self._preview = QLabel()
        self._preview.setMinimumSize(200, 200)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet(f"background:{BG_CARD}; border:1px solid {BORDER};")
        prev_col.addWidget(self._preview, 1)
        mid.addLayout(left_col, 4)
        mid.addLayout(list_col, 0)
        mid.addLayout(prev_col, 3)
        root.addLayout(mid, 1)
        st = QHBoxLayout()
        self._status = QLabel()
        self._status.setObjectName("statusLabel")
        self._dirty_lbl = QLabel("")
        self._dirty_lbl.setStyleSheet(f"color:{ORANGE}; font-weight:bold;")
        st.addWidget(self._status, 1)
        st.addWidget(self._dirty_lbl)
        sb = QWidget()
        sb.setLayout(st)
        sb.setStyleSheet(f"background:{BG_PANEL};")
        root.addWidget(sb)

    @staticmethod
    def _bar(color: str) -> QWidget:
        w = QWidget()
        w.setFixedSize(3, 14)
        w.setStyleSheet(f"background:{color};")
        return w

    def _hdr_row(self, title: str, color: str) -> QHBoxLayout:
        h = QHBoxLayout()
        h.addWidget(self._bar(color))
        h.addWidget(QLabel(f"<b>{title}</b>"))
        h.addStretch(1)
        return h

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
        self.status_changed.emit(text)

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

    def _choose_labels_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select labels directory", str(self.labels_root.parent))
        if not folder:
            return
        self.labels_root = Path(folder).resolve()
        self._labels_dir_lbl.setText(f"Labels: {self.labels_root}")
        self._reload_current()

    def open_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select images folder", str(self.images_root))
        if not selected:
            return
        base = Path(selected).resolve()
        images = [p for p in sorted(base.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not images:
            QMessageBox.warning(self, "No images", "No supported images found.")
            return
        self.image_paths = images
        self.images_root = base
        self.index = 0
        self.new_points_canvas = []
        self._auto_detect_labels_dir()
        self._reload_current()

    def open_specific_image(self, image_path: Path, labels_root: Path | None = None) -> None:
        target = image_path.resolve()
        images_root = target.parent
        if target.parent.name == "images":
            images_root = target.parent
            if labels_root is None:
                labels_root = target.parent.parent / "labels"
        self.images_root = images_root
        images = [p for p in sorted(images_root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not images:
            QMessageBox.warning(self, "No images", f"No supported images found in {images_root}.")
            return
        self.image_paths = images
        try:
            self.index = self.image_paths.index(target)
        except ValueError:
            self.index = 0
        if labels_root is not None:
            self.labels_root = labels_root.resolve()
            self._labels_dir_lbl.setText(f"Labels: {self.labels_root}")
        else:
            self._auto_detect_labels_dir()
        self.new_points_canvas = []
        self._reload_current()

    def _auto_detect_labels_dir(self) -> None:
        base = self.images_root
        if base.name == "images" and (base.parent / "labels").is_dir():
            self.labels_root = (base.parent / "labels").resolve()
            self._labels_dir_lbl.setText(f"Labels: {self.labels_root}")
            return
        parent = base.parent
        for candidate in [parent / "labels", parent / "labels_poly"]:
            if candidate.is_dir():
                self.labels_root = candidate.resolve()
                self._labels_dir_lbl.setText(f"Labels: {self.labels_root}")
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

    def prev_image(self) -> None:
        if not self.image_paths:
            return
        self.index = max(0, self.index - 1)
        self.new_points_canvas = []
        self._reload_current()

    def next_image(self) -> None:
        if not self.image_paths:
            return
        self.index = min(len(self.image_paths) - 1, self.index + 1)
        self.new_points_canvas = []
        self._reload_current()

    def _reload_current(self) -> None:
        self.pois = []
        self.selected_poi = -1
        self.dragging = None
        self._zoom_reset()
        self._mark_clean()
        self._load_image()
        self._load_pois_for_current()
        self._rebuild_poi_list()
        self.render()

    def _load_image(self) -> None:
        if not self.image_paths:
            self.current_image = None
            return
        image_path = self.image_paths[self.index]
        img = cv2.imread(str(image_path))
        if img is None:
            self.current_image = None
            self._set_status(f"Failed to read {image_path.name}")
            return
        self.current_image = img

    def _rebuild_poi_list(self) -> None:
        self._poi_list.clear()
        for i, poi in enumerate(self.pois):
            item = QListWidgetItem(f"POI {i + 1} ({len(poi.points)} pts)")
            if i == self.selected_poi:
                item.setBackground(QColor(BG_CARD))
                item.setForeground(QColor(ACCENT))
            self._poi_list.addItem(item)
        if not self.pois:
            self._poi_list.addItem("(none — Add POI or A)")

    def _on_poi_list_click(self, item: QListWidgetItem) -> None:
        row = self._poi_list.row(item)
        if 0 <= row < len(self.pois):
            self.selected_poi = row
            self.new_points_canvas = []
            self._rebuild_poi_list()
            self.render()

    def _load_pois_for_current(self) -> None:
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

    def _auto_save(self) -> None:
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
        elif label_path.exists():
            label_path.unlink()

    def save_all(self) -> None:
        self._mark_dirty()
        if not self.image_paths or self.current_image is None:
            return
        image_path = self.image_paths[self.index]
        label_path = self._label_path_for(image_path)
        try:
            export_fn = self._save_to_train_callback or self._default_export_to_train
            msg = export_fn(image_path, label_path if label_path.exists() else None)
            if msg:
                self._set_status(f"{self._status.text()} | {msg}")
        except Exception as exc:
            self._set_status(f"{self._status.text()} | Train export failed: {exc}")

    def delete_label_file(self) -> None:
        if not self.image_paths:
            return
        image_path = self.image_paths[self.index]
        label_path = self._label_path_for(image_path)
        if not label_path.exists():
            self._set_status(f"No label file: {image_path.name}")
            return
        if QMessageBox.question(
            self, "Delete label file",
            f"Permanently delete label for {image_path.name}?\n\n{label_path}",
        ) != QMessageBox.StandardButton.Yes:
            return
        label_path.unlink()
        self.pois = []
        self.selected_poi = -1
        self.new_points_canvas = []
        self._mark_dirty()
        self._rebuild_poi_list()
        self.render()
        self._set_status(f"Deleted label file for {image_path.name}")

    def _run_ai_assist(self) -> None:
        if not self.image_paths or self.current_image is None:
            self._set_status("AI Assist: No image loaded.")
            return
        if not self._ai_assist_callback:
            self._set_status("AI Assist: Not connected to model.")
            return
        self._set_status("AI Assist: Predicting...")
        try:
            points_list = self._ai_assist_callback(self.current_image)
            if not points_list:
                self._set_status("AI Assist: No POIs found.")
                return
            self.pois = []
            for i, pts in enumerate(points_list):
                self.pois.append(EditablePoi(pts, color_idx=i))
            self.selected_poi = 0
            self._rebuild_poi_list()
            self.render()
            self._mark_dirty()
            self._set_status(f"AI Assist: Auto-populated {len(points_list)} POIs.")
        except Exception as e:
            self._set_status(f"AI Assist Error: {e}")

    def _start_add_poi(self) -> None:
        self.new_points_canvas = []
        self.selected_poi = -1
        self._rebuild_poi_list()
        self._set_status("Click 4 corners for new POI. Enter to confirm, Esc to cancel.")
        self.render()

    def _cancel_add(self) -> None:
        self.new_points_canvas = []
        self._set_status("Cancelled.")
        self.render()

    def _delete_selected_poi(self) -> None:
        if self.selected_poi < 0 or self.selected_poi >= len(self.pois):
            self._set_status("Select a POI first.")
            return
        self.pois.pop(self.selected_poi)
        for i, poi in enumerate(self.pois):
            poi.color_idx = i
        self.selected_poi = min(self.selected_poi, len(self.pois) - 1)
        self._mark_dirty()
        self._rebuild_poi_list()
        self.render()
        self._set_status(f"Deleted. {len(self.pois)} POI(s) remaining.")

    def _zoom_reset(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _zoom_in(self) -> None:
        self._apply_zoom_center(self._ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._apply_zoom_center(1.0 / self._ZOOM_STEP)

    def _apply_zoom_center(self, factor: float) -> None:
        cw, ch = max(1, self._canvas.width()), max(1, self._canvas.height())
        self._zoom_at(cw / 2, ch / 2, factor)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        old_zoom = self._zoom
        new_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, self._zoom * factor))
        if new_zoom == old_zoom:
            return
        base_x = (cx - self._pan_x) / old_zoom
        base_y = (cy - self._pan_y) / old_zoom
        self._pan_x = cx - base_x * new_zoom
        self._pan_y = cy - base_y * new_zoom
        self._zoom = new_zoom
        self.render()

    def undo_point(self) -> None:
        if self.new_points_canvas:
            self.new_points_canvas.pop()
            self.render()

    def clear_points(self) -> None:
        self.new_points_canvas = []
        self.render()

    def _fit_image_zoomed_widget(self, bgr: np.ndarray, cw: int, ch: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        w, h = image.size
        base_scale = min(cw / w, ch / h)
        eff_scale = base_scale * self._zoom
        nw = max(1, int(w * eff_scale))
        nh = max(1, int(h * eff_scale))
        fitted = image.resize((nw, nh), Image.Resampling.BILINEAR)
        base_x = (cw - int(w * base_scale)) // 2
        base_y = (ch - int(h * base_scale)) // 2
        zoom_cx, zoom_cy = cw / 2, ch / 2
        final_x = zoom_cx - (zoom_cx - base_x) * self._zoom + self._pan_x
        final_y = zoom_cy - (zoom_cy - base_y) * self._zoom + self._pan_y
        return fitted, final_x, final_y, nw, nh

    def render(self) -> None:
        if not self.image_paths or self.current_image is None:
            if not self.image_paths:
                self._set_status("Open an images folder to begin.")
            self._canvas.update()
            return
        cw, ch = max(1, self._canvas.width()), max(1, self._canvas.height())
        fitted, x, y, nw, nh = self._fit_image_zoomed_widget(self.current_image, cw, ch)
        self.render_info = {"x": x, "y": y, "w": nw, "h": nh}
        image_path = self.image_paths[self.index]
        n_pois = len(self.pois)
        sel_txt = f" | Selected: POI {self.selected_poi + 1}" if self.selected_poi >= 0 else ""
        add_txt = f" | Adding: {len(self.new_points_canvas)}/4" if self.new_points_canvas else ""
        zoom_txt = f" | Zoom: {int(self._zoom * 100)}%" if self._zoom != 1.0 else ""
        self._set_status(
            f"{self.index + 1}/{len(self.image_paths)} | {image_path.name} | "
            f"{n_pois} POI(s){sel_txt}{add_txt}{zoom_txt} | Scroll=zoom Mid-drag=pan"
        )
        self._canvas.update()
        self._render_preview()

    def _canvas_to_image(self, cx: int, cy: int) -> tuple[int, int]:
        if self.current_image is None:
            return 0, 0
        ih, iw = self.current_image.shape[:2]
        rx, ry, rw, rh = self.render_info["x"], self.render_info["y"], self.render_info["w"], self.render_info["h"]
        nx = (cx - rx) / float(max(1, rw))
        ny = (cy - ry) / float(max(1, rh))
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        return int(nx * (iw - 1)), int(ny * (ih - 1))

    def _image_to_canvas(self, ix: int, iy: int) -> tuple[float, float]:
        if self.current_image is None:
            return 0.0, 0.0
        ih, iw = self.current_image.shape[:2]
        rx, ry, rw, rh = self.render_info["x"], self.render_info["y"], self.render_info["w"], self.render_info["h"]
        cx = rx + (ix / max(1, iw - 1)) * rw
        cy = ry + (iy / max(1, ih - 1)) * rh
        return cx, cy

    def _draw_poi_painter(self, painter: QPainter, poi: EditablePoi, color: QColor, selected: bool) -> None:
        if len(poi.points) < 2:
            return
        poly = QPolygonF([QPointF(*self._image_to_canvas(px, py)) for px, py in poi.points])
        if len(poi.points) >= 3:
            c = QColor(color)
            c.setAlpha(60 if selected else 40)
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(poly)
        pen = QPen(color, 3 if selected else 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)
        for pt_idx, (px, py) in enumerate(poi.points):
            cx, cy = self._image_to_canvas(px, py)
            r = 5 if selected else 3
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#000000") if selected else color))
            painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
            if selected:
                painter.setPen(QColor(color))
                painter.drawText(int(cx + 8), int(cy - 8), str(pt_idx + 1))

    def _draw_new_points_painter(self, painter: QPainter) -> None:
        color = QColor("#00ff66")
        prev = None
        for idx, (ix, iy) in enumerate(self.new_points_canvas):
            px, py = self._image_to_canvas(ix, iy)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#000000")))
            painter.drawEllipse(QRectF(px - 5, py - 5, 10, 10))
            painter.setPen(QPen(color))
            painter.drawText(int(px + 10), int(py - 10), str(idx + 1))
            if prev:
                painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
                painter.drawLine(prev[0], prev[1], px, py)
            prev = (px, py)

    def _render_preview(self) -> None:
        if self.current_image is None:
            self._preview.clear()
            return
        preview = self.current_image.copy()
        for i, poi in enumerate(self.pois):
            if len(poi.points) < 3:
                continue
            bgr = POI_COLORS_BGR[i % len(POI_COLORS_BGR)]
            pts_np = np.array(poi.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(preview, [pts_np], bgr)
            cv2.polylines(preview, [pts_np], True, bgr, 2, cv2.LINE_AA)
            for px, py in poi.points:
                cv2.circle(preview, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in poi.points]))
            cy = int(np.mean([p[1] for p in poi.points]))
            cv2.putText(preview, f"POI {i + 1}", (cx - 20, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
        h, w = preview.shape[:2]
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        pix = QPixmap.fromImage(
            QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        )
        pix = pix.scaled(self._preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._preview.setPixmap(pix)

    def _canvas_mouse_press(self, e) -> None:
        if self.current_image is None:
            return
        if e.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = (e.position().x(), e.position().y())
            self._pan_start_offset = (self._pan_x, self._pan_y)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        ex, ey = int(e.position().x()), int(e.position().y())
        for poi_idx, poi in enumerate(self.pois):
            for pt_idx, (px, py) in enumerate(poi.points):
                cx, cy = self._image_to_canvas(px, py)
                if abs(ex - cx) <= DRAG_RADIUS and abs(ey - cy) <= DRAG_RADIUS:
                    self.dragging = (poi_idx, pt_idx)
                    self.selected_poi = poi_idx
                    self._mark_dirty()
                    self._rebuild_poi_list()
                    return
        ix, iy = self._canvas_to_image(ex, ey)
        for poi_idx, poi in enumerate(self.pois):
            if len(poi.points) >= 3:
                pts_np = np.array(poi.points, dtype=np.int32)
                if cv2.pointPolygonTest(pts_np, (ix, iy), False) >= 0:
                    self.selected_poi = poi_idx
                    self.new_points_canvas = []
                    self._rebuild_poi_list()
                    self.render()
                    return
        ix_clamped = max(0, min(self.current_image.shape[1] - 1, ix))
        iy_clamped = max(0, min(self.current_image.shape[0] - 1, iy))
        self.new_points_canvas.append((ix_clamped, iy_clamped))
        if len(self.new_points_canvas) == 4:
            self._finish_add_if_ready()
        else:
            self._mark_dirty()
            self._set_status(f"New POI: {len(self.new_points_canvas)}/4 corners.")
        self.render()

    def _canvas_mouse_move(self, e) -> None:
        if self._panning and self._pan_start:
            dx = e.position().x() - self._pan_start[0]
            dy = e.position().y() - self._pan_start[1]
            self._pan_x = self._pan_start_offset[0] + dx
            self._pan_y = self._pan_start_offset[1] + dy
            self.render()
            return
        if self.dragging is None or self.current_image is None:
            return
        poi_idx, pt_idx = self.dragging
        ix, iy = self._canvas_to_image(int(e.position().x()), int(e.position().y()))
        h, w = self.current_image.shape[:2]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        self.pois[poi_idx].points[pt_idx] = (ix, iy)
        self.render()

    def _canvas_mouse_release(self, e) -> None:
        if e.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_start = None
            return
        if self.dragging is not None:
            self.dragging = None
            self._mark_dirty()
            self._set_status("Point moved.")

    def _canvas_wheel(self, e) -> None:
        delta = e.angleDelta().y()
        factor = self._ZOOM_STEP if delta > 0 else 1.0 / self._ZOOM_STEP
        self._zoom_at(e.position().x(), e.position().y(), factor)

    def _canvas_double_click(self, _e) -> None:
        self._zoom_reset()
        self.new_points_canvas = []
        self.render()

    def _mark_dirty(self) -> None:
        self._auto_save()
        self._dirty_lbl.setText("✓ saved")
        if self._dirty_timer is None:
            self._dirty_timer = QTimer(self)
            self._dirty_timer.setSingleShot(True)
            self._dirty_timer.timeout.connect(self._mark_clean)
        self._dirty_timer.start(2000)

    def _mark_clean(self) -> None:
        self._dirty_lbl.setText("")

    def _finish_add_if_ready(self) -> None:
        if len(self.new_points_canvas) == 4:
            color_idx = len(self.pois) % len(POI_COLORS_HEX)
            self.pois.append(EditablePoi(list(self.new_points_canvas), color_idx))
            self.selected_poi = len(self.pois) - 1
            self.new_points_canvas = []
            self._mark_dirty()
            self._rebuild_poi_list()
            self._set_status(f"Added POI {len(self.pois)}")
            self.render()

    def handle_shortcut_prev_next(self, prev: bool) -> None:
        if prev:
            self.prev_image()
        else:
            self.next_image()

    def close_cleanup(self) -> None:
        pass
