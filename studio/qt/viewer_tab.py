"""Run tab: inference viewer (PySide6)."""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageGrab
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from studio.config import (
    AUTOLABEL_MIN_INTERVAL,
    CAM_DISCONNECT_THRESHOLD,
    DATA_AUTOLABEL_DIR,
    FG_DIM,
    LIVE_TARGET_MS,
    MONO,
    ORANGE,
    POI_COLORS,
    POI_COLORS_HEX,
    RUNS_DIR,
    SUPPORTED_EXTENSIONS,
    _select_device,
    resolve_model_path,
)
from studio.qt.images import numpy_bgr_to_qpixmap
from studio.qt.widgets import (
    BTN_HEIGHT,
    make_button,
    make_menu_button,
    normalize_toolbar_widget,
    toolbar_panel,
    toolbar_row,
    toolbar_separator,
)
from studio.qt.win32 import apply_capture_exclusion_for_hwnd


class ViewerTab(QWidget):
    """Desktop inference viewer embedded in main window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device = _select_device()
        resolved = resolve_model_path()
        self.model_path = resolved if resolved is not None else Path("best.pt")
        self.model: YOLO | None = None
        self.current_image_path: Path | None = None
        self.image_paths: list[Path] = []
        self.index = 0

        self.live_running = False
        self.live_timer: QTimer | None = None
        self.camera_capture: cv2.VideoCapture | None = None
        self.active_live_source: str | None = None
        self._camera_lock = threading.Lock()
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._camera_reader_thread: threading.Thread | None = None
        self._camera_reader_stop = threading.Event()
        self._camera_dead = False
        self._camera_results_queue: queue.Queue[list[str]] = queue.Queue()
        self._live_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._screen_excluded = False
        self._inference_pending = False
        self._capture_for_relabel_callback = None

        self.autolabel_active = False
        self.autolabel_dir = DATA_AUTOLABEL_DIR
        self._autolabel_count = 0
        self._autolabel_last_save = 0.0
        self._last_infer_frame: np.ndarray | None = None
        self._last_infer_result = None

        self._build_ui()
        self._load_model()
        self._refresh_camera_list_background()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        title = QLabel(
            "<span style='color:#00a86b;font-weight:bold;font-size:12pt'>POI</span> "
            "<span style='font-weight:bold;font-size:12pt'>Viewer</span>"
        )
        normalize_toolbar_widget(title)

        self._source_combo = QComboBox()
        self._source_combo.addItems(["Camera", "Screen"])
        self._source_combo.setToolTip("Live input: webcam or full screen")
        self._source_combo.currentTextChanged.connect(self._on_live_source_changed)
        normalize_toolbar_widget(self._source_combo)
        self._cam_combo = QComboBox()
        self._cam_combo.addItem("0")
        self._cam_combo.setToolTip("Camera device index")
        self._cam_combo.setMinimumWidth(56)
        normalize_toolbar_widget(self._cam_combo)
        live_lbl = QLabel("Live:")
        normalize_toolbar_widget(live_lbl)

        self._btn_start = make_button(
            "▶ Start live", self._start_live, style="primary",
            tooltip="Start live inference (camera or screen)",
        )
        self._btn_stop = make_button(
            "■ Stop", self._stop_live, style="danger",
            tooltip="Stop live capture and inference",
        )
        self._btn_autolabel = make_button(
            "Auto-label: OFF", self._toggle_autolabel,
            tooltip="Automatically save frames with POI detections (Space)",
        )
        self._btn_autolabel.setMinimumWidth(130)

        row1 = toolbar_row(
            title,
            toolbar_separator(),
            make_menu_button("File ▾", [
                ("Open image…", self._open_image),
                ("Open folder…", self._open_folder)
            ], tooltip="Open an image or folder"),
            make_menu_button("Model ▾", [
                ("Choose model .pt…", self._choose_model),
                ("Load latest from runs/", self._load_last_trained)
            ], tooltip="Load YOLO weights"),
            toolbar_separator(),
            make_button("◀", self._prev_image, style="compact", tooltip="Previous image (←)"),
            make_button("▶", self._next_image, style="compact", tooltip="Next image (→)"),
            toolbar_separator(),
            live_lbl,
            self._source_combo,
            self._cam_combo,
            make_button("↻", self._refresh_camera_list_background, style="compact",
                        tooltip="Scan for connected cameras"),
            self._btn_start,
            self._btn_stop,
            toolbar_separator(),
            self._btn_autolabel,
            make_menu_button("Capture ▾", [
                ("Send current frame to Relabel", self._capture_for_relabel),
                ("-", None),
                ("Set auto-label output folder…", self._choose_autolabel_dir)
            ], tooltip="Capture current frame or set output dir"),
        )

        footer_row = QWidget()
        fr = QHBoxLayout(footer_row)
        fr.setContentsMargins(0, 0, 0, 0)
        self._autolabel_counter_lbl = QLabel("")
        self._autolabel_counter_lbl.setStyleSheet(f"color: {ORANGE}; font-weight: bold;")
        fr.addWidget(self._autolabel_counter_lbl)
        fr.addStretch(1)
        dev = QLabel(f"Device: {self.device.upper()}")
        dev.setStyleSheet(
            "background:#222;color:#00a86b;font-family:Consolas;font-size:9pt;"
            "padding:4px 10px;border-radius:6px;border:1px solid #333;"
        )
        normalize_toolbar_widget(dev, height=BTN_HEIGHT - 4)
        fr.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        fr.addWidget(dev)

        root.addWidget(toolbar_panel(row1, footer_row))

        # Viewport
        mid = QHBoxLayout()
        mid.setContentsMargins(12, 8, 12, 4)

        left_wrap = QVBoxLayout()
        left_hdr = QHBoxLayout()
        left_hdr.addWidget(self._accent_bar())
        left_hdr.addWidget(QLabel("<b>Image + POI Overlay</b>"))
        left_hdr.addStretch(1)
        left_wrap.addLayout(left_hdr)
        self._left_label = QLabel()
        self._left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._left_label.setMinimumSize(320, 240)
        self._left_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._left_label.setStyleSheet("background:#222;border:1px solid #333;")
        left_wrap.addWidget(self._left_label, 1)

        right_wrap = QVBoxLayout()
        rh = QHBoxLayout()
        rh.addWidget(self._accent_bar())
        rh.addWidget(QLabel("<b>POI Crops</b>"))
        rh.addStretch(1)
        right_wrap.addLayout(rh)
        self._crops_scroll = QScrollArea()
        self._crops_scroll.setWidgetResizable(True)
        self._crops_inner = QWidget()
        self._crops_inner_layout = QGridLayout(self._crops_inner)
        self._crops_scroll.setWidget(self._crops_inner)
        self._crops_scroll.setMinimumWidth(280)
        self._crops_scroll.setStyleSheet("background:#222;border:1px solid #333;")
        right_wrap.addWidget(self._crops_scroll, 1)

        mid.addLayout(left_wrap, 2)
        mid.addLayout(right_wrap, 1)
        mid_w = QWidget()
        mid_w.setLayout(mid)
        root.addWidget(mid_w, 1)

        # Status
        st = QHBoxLayout()
        st.setContentsMargins(14, 6, 14, 6)
        self._status = QLabel("Ready")
        self._status.setObjectName("statusLabel")
        st.addWidget(self._status, 1)
        self._autolabel_path_lbl = QLabel(f"📂 {self.autolabel_dir}")
        self._autolabel_path_lbl.setObjectName("statusLabel")
        st.addWidget(self._autolabel_path_lbl)
        sb = QWidget()
        sb.setLayout(st)
        sb.setStyleSheet("background:#1a1a1a;")
        root.addWidget(sb)

    @staticmethod
    def _accent_bar() -> QWidget:
        w = QWidget()
        w.setFixedSize(3, 14)
        w.setStyleSheet("background:#00a86b;")
        return w

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_capture_for_relabel_callback(self, cb) -> None:
        self._capture_for_relabel_callback = cb

    def _hwnd(self) -> int:
        w = self.window()
        return int(w.winId()) if w else 0

    def _apply_screen_exclusion(self) -> None:
        hwnd = self._hwnd()
        ok = apply_capture_exclusion_for_hwnd(hwnd, True)
        self._screen_excluded = ok
        if ok:
            self._set_status("Screen exclusion active — app hidden from capture")
        else:
            self._set_status("Screen exclusion not available (requires Windows 10 2004+)")

    def _remove_screen_exclusion(self) -> None:
        if self._screen_excluded:
            apply_capture_exclusion_for_hwnd(self._hwnd(), False)
            self._screen_excluded = False

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self._set_status(f"Model not found: {self.model_path}  |  device={self.device}")
            self.model = None
            return
        try:
            self.model = YOLO(str(self.model_path))
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.predict(source=dummy, device=self.device, conf=0.50, imgsz=960, verbose=False)
            self._set_status(f"Loaded {self.model_path.name}  |  device={self.device}")
        except Exception as exc:
            self.model = None
            self._set_status(f"Failed to load model: {exc}")

    def load_trained_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self._set_status(f"Model not found: {path}")
            return
        self.model_path = p
        self._load_model()
        if self.current_image_path and self.model is not None:
            self._run_prediction(self.current_image_path)

    def _choose_model(self) -> None:
        self._stop_live()
        path, _ = QFileDialog.getOpenFileName(self, "Choose YOLO model", "", "PyTorch (*.pt);;All (*.*)")
        if not path:
            return
        self.model_path = Path(path)
        self._load_model()
        if self.current_image_path and self.model is not None:
            self._run_prediction(self.current_image_path)

    def _open_image(self) -> None:
        self._stop_live()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All (*.*)",
        )
        if not path:
            return
        image_path = Path(path)
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            QMessageBox.critical(self, "Unsupported file", f"Unsupported extension: {image_path.suffix}")
            return
        self.image_paths = []
        self.current_image_path = image_path
        self._run_prediction(image_path)

    def _open_folder(self) -> None:
        self._stop_live()
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        base = Path(folder)
        images = sorted(
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            QMessageBox.warning(self, "No images", "No supported images found in folder.")
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
        self.current_image_path = self.image_paths[self.index]
        self._run_prediction(self.current_image_path)

    def _run_prediction(self, image_path: Path) -> None:
        if self.model is None:
            QMessageBox.critical(self, "Model not loaded", "Load a valid .pt model first.")
            return
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            QMessageBox.critical(self, "Image error", f"Could not read: {image_path}")
            return
        tag = f"[{self.index + 1}/{len(self.image_paths)}]" if self.image_paths else ""
        self._set_status(f"{tag} Predicting: {image_path.name} ...")
        try:
            result = self.model.predict(source=bgr, device=self.device, conf=0.50, imgsz=960, verbose=False)[0]
        except Exception as exc:
            QMessageBox.critical(self, "Prediction error", str(exc))
            return
        overlay, crops, status = self._build_outputs(bgr, result)
        self._last_infer_frame = bgr.copy()
        self._last_infer_result = result
        self._set_status(status)
        self._set_preview_images(overlay, crops)

    def _open_camera_capture(self) -> bool:
        try:
            cam_index = int(self._cam_combo.currentText().strip())
        except ValueError:
            QMessageBox.critical(self, "Camera index", "Camera index must be an integer.")
            return False
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cam_index, backend)
        if not cap.isOpened():
            QMessageBox.critical(self, "Camera error", f"Could not open camera index {cam_index}.")
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
        self._camera_reader_thread = threading.Thread(target=self._camera_reader_loop, daemon=True)
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
        self._set_status("Scanning for cameras...")

        def _worker() -> None:
            self._camera_results_queue.put(self._detect_camera_indices())

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(50, self._poll_camera_scan_results)

    def _poll_camera_scan_results(self) -> None:
        latest: list[str] | None = None
        try:
            while True:
                latest = self._camera_results_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            QTimer.singleShot(50, self._poll_camera_scan_results)
            return
        self._apply_camera_list(latest)

    def _apply_camera_list(self, cameras: list[str]) -> None:
        if not cameras:
            cameras = ["0"]
            self._set_status("No camera detected. Using default index 0.")
        else:
            self._set_status(f"Found cameras: {', '.join(cameras)}")
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for c in cameras:
            self._cam_combo.addItem(c)
        self._cam_combo.blockSignals(False)

    def _on_live_source_changed(self, _text: str) -> None:
        if not self.live_running:
            return
        new_source = self._source_combo.currentText()
        if new_source == self.active_live_source:
            return
        self._stop_camera_reader()
        self._close_camera()
        self._remove_screen_exclusion()
        if new_source == "Camera":
            if not self._open_camera_capture():
                self._set_status("Camera open failed after source switch")
            else:
                self._start_camera_reader()
        elif new_source == "Screen":
            self._apply_screen_exclusion()
        self.active_live_source = new_source

    def _start_live(self) -> None:
        if self.model is None:
            QMessageBox.critical(self, "Model not loaded", "Load a valid .pt model first.")
            return
        if self.live_running:
            return
        source = self._source_combo.currentText()
        if source == "Camera":
            if not self._open_camera_capture():
                return
            self._start_camera_reader()
        elif source == "Screen":
            self._apply_screen_exclusion()
        self.active_live_source = source
        self._inference_pending = False
        self.live_running = True
        if self.live_timer is None:
            self.live_timer = QTimer(self)
            self.live_timer.timeout.connect(self._live_tick)
        self.live_timer.setInterval(max(1, LIVE_TARGET_MS))
        self.live_timer.start()

    def _stop_live(self) -> None:
        self.live_running = False
        if self.live_timer is not None:
            self.live_timer.stop()
        self._stop_camera_reader()
        self._close_camera()
        self._remove_screen_exclusion()
        self.active_live_source = None
        self._inference_pending = False

    def stop_live_for_tab_switch(self) -> None:
        self._stop_live()

    def _read_live_frame(self) -> tuple[np.ndarray | None, str]:
        source = self.active_live_source or self._source_combo.currentText()
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
        return frame.copy(), f"Live camera {self._cam_combo.currentText().strip()}"

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
            return
        frame, label = self._read_live_frame()
        if frame is not None:
            frame_copy = frame.copy()
            self._inference_pending = True
            model = self.model

            def _worker() -> None:
                try:
                    assert model is not None
                    result = model.predict(
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
            self._set_status(label)
            if self._camera_dead:
                self._stop_live()
                return
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        delay = max(1, LIVE_TARGET_MS - int(elapsed_ms))
        if self.live_timer is not None and self.live_running:
            self.live_timer.setInterval(delay)

    def _deliver_live_result(
        self, overlay: np.ndarray, crops: list, status: str,
        raw_frame: np.ndarray | None, result,
    ) -> None:
        self._inference_pending = False
        if not self.live_running:
            return
        self._set_preview_images(overlay, crops)
        self._set_status(status)
        self._last_infer_frame = raw_frame.copy() if raw_frame is not None else None
        self._last_infer_result = result
        if self.autolabel_active and raw_frame is not None and result is not None:
            self._autolabel_save(raw_frame, result)

    def _deliver_live_error(self, error: str) -> None:
        self._inference_pending = False
        if not self.live_running:
            return
        self._set_status(f"Prediction error: {error}")

    def _toggle_autolabel(self) -> None:
        if self.autolabel_active:
            self.autolabel_active = False
            self._btn_autolabel.setText("Auto-label: OFF")
            self._btn_autolabel.setObjectName("secondaryButton")
            self._btn_autolabel.style().unpolish(self._btn_autolabel)
            self._btn_autolabel.style().polish(self._btn_autolabel)
            self._set_status("Auto-label stopped")
        else:
            self.autolabel_dir.mkdir(parents=True, exist_ok=True)
            (self.autolabel_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.autolabel_dir / "labels").mkdir(parents=True, exist_ok=True)
            self.autolabel_active = True
            self._autolabel_last_save = 0.0
            self._btn_autolabel.setText("Auto-label: ON")
            self._btn_autolabel.setObjectName("warningButton")
            self._btn_autolabel.style().unpolish(self._btn_autolabel)
            self._btn_autolabel.style().polish(self._btn_autolabel)
            self._update_autolabel_counter()
            self._set_status(f"Auto-label active → {self.autolabel_dir}")
        self._autolabel_path_lbl.setText(f"📂 {self.autolabel_dir}")

    def _choose_autolabel_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose auto-label output", str(self.autolabel_dir.parent))
        if not folder:
            return
        self.autolabel_dir = Path(folder)
        self._autolabel_count = 0
        self._update_autolabel_counter()
        self._autolabel_path_lbl.setText(f"📂 {self.autolabel_dir}")
        self._set_status(f"Auto-label output set to {self.autolabel_dir}")

    def _autolabel_save(self, bgr: np.ndarray, result) -> None:
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
            for corners in obb.xyxyxyxy.cpu().numpy():
                quad = self._order_quad_points(corners.astype(np.float32))
                coords: list[str] = []
                for px, py in quad:
                    coords.append(f"{px / w:.6f}")
                    coords.append(f"{py / h:.6f}")
                lines.append("0 " + " ".join(coords))
        lbl_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        return img_path, lbl_path

    def _capture_for_relabel(self) -> None:
        frame = self._last_infer_frame
        result = self._last_infer_result
        if frame is None or result is None:
            self._set_status("Nothing to capture yet. Run one prediction first.")
            return
        saved = self._save_prediction_as_label(frame, result)
        if not saved:
            self._set_status("Capture failed: could not save current frame.")
            return
        img_path, _ = saved
        self._autolabel_count += 1
        self._update_autolabel_counter()
        self._set_status(f"Captured for relabel: {img_path.name}")
        if self._capture_for_relabel_callback:
            self._capture_for_relabel_callback(img_path, self.autolabel_dir / "labels")

    def _update_autolabel_counter(self) -> None:
        self._autolabel_counter_lbl.setText(f"💾 {self._autolabel_count}" if self._autolabel_count else "")

    def toggle_autolabel_shortcut(self) -> None:
        self._toggle_autolabel()

    def _build_outputs(self, bgr: np.ndarray, result) -> tuple[np.ndarray, list[dict], str]:
        overlay = bgr.copy()
        h, w = bgr.shape[:2]
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            cv2.putText(overlay, "POI not detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            status = "POI not detected"
            if self.image_paths:
                status = f"[{self.index + 1}/{len(self.image_paths)}] {status}"
            return overlay, [], status

        all_crops: list[dict] = []
        rect_summaries: list[str] = []
        all_corners = obb.xyxyxyxy.cpu().numpy()
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
            cv2.putText(overlay, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
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

    def _set_preview_images(self, overlay_bgr: np.ndarray, crops_data: list[dict]) -> None:
        pm = numpy_bgr_to_qpixmap(overlay_bgr)
        pm = pm.scaled(660, 660, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._left_label.setPixmap(pm)
        while self._crops_inner_layout.count():
            item = self._crops_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not crops_data:
            lbl = QLabel("No POI detected")
            lbl.setStyleSheet(f"color: {FG_DIM}; padding: 40px;")
            self._crops_inner_layout.addWidget(lbl, 0, 0)
            return
        cols = 2
        for idx, crop_info in enumerate(crops_data):
            row, col = divmod(idx, cols)
            card = QWidget()
            card.setStyleSheet(f"background:#2a2a2a; border:1px solid #333;")
            vl = QVBoxLayout(card)
            hdr = QLabel(f"  {crop_info['label']}   ({crop_info['bbox'][0]},{crop_info['bbox'][1]})-({crop_info['bbox'][2]},{crop_info['bbox'][3]})")
            hdr.setStyleSheet(f"background:{crop_info['color_hex']}; color:#000; font-weight:bold;")
            vl.addWidget(hdr)
            crop_bgr = crop_info["image"]
            ch, cw = crop_bgr.shape[:2]
            target_size = 250
            scale = min(target_size / max(cw, 1), target_size / max(ch, 1), 1.0)
            dw, dh = max(1, int(cw * scale)), max(1, int(ch * scale))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            resized = cv2.resize(crop_bgr, (dw, dh), interpolation=interp)
            il = QLabel()
            il.setPixmap(numpy_bgr_to_qpixmap(resized))
            vl.addWidget(il)
            self._crops_inner_layout.addWidget(card, row, col)

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

    def _load_last_trained(self) -> None:
        self._stop_live()
        try:
            candidates = list(RUNS_DIR.rglob("best.pt"))
            if not candidates:
                QMessageBox.information(self, "No Models", "No trained models found in runs/ directory.")
                return
            last_model = max(candidates, key=lambda p: p.stat().st_mtime)
            self.model_path = last_model
            self._load_model()
            self._set_status(f"Loaded last trained model: {last_model.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to find last trained model: {e}")

    def predict_for_relabel(self, bgr_image: np.ndarray) -> list[list[tuple[int, int]]]:
        if not self.model:
            raise RuntimeError("YOLO model is not loaded on the Run tab.")
        result = self.model.predict(
            source=bgr_image, device=self.device, conf=0.50, imgsz=960, verbose=False,
        )[0]
        points_list: list[list[tuple[int, int]]] = []
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            for corners in obb.xyxyxyxy.cpu().numpy():
                quad = self._order_quad_points(corners.astype(np.float32))
                pts = [(int(x), int(y)) for x, y in quad]
                points_list.append(pts)
        return points_list

    def close_cleanup(self) -> None:
        self._stop_live()
        self._remove_screen_exclusion()
