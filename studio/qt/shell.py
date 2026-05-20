"""Main window: Run + Data + Train (PySide6)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from studio.config import BG, BG_PANEL, DATA_LABELED_DIR
from studio.qt.data_tab import DataTab
from studio.qt.theme import application_stylesheet
from studio.qt.train_tab import TrainTab
from studio.qt.viewer_tab import ViewerTab
from studio.qt.win32 import apply_capture_exclusion_for_hwnd


class PoiStudioWindow(QWidget):
    """Unified POI Studio (tabs: Run, Data, Train)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("POI Studio - Detection Platform")
        self.resize(1500, 920)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(f"background:{BG};")

        self._current_tab = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tab_bar = QWidget()
        tab_bar.setObjectName("topTabBar")
        tb_lay = QHBoxLayout(tab_bar)
        tb_lay.setContentsMargins(8, 6, 8, 6)
        tb_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._tab_buttons: list[QPushButton] = []
        for i, name in enumerate(["Run", "Data", "Train"]):
            b = QPushButton(name)
            b.setObjectName("tabButton")
            b.setFixedHeight(40)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"Switch to {name} tab")
            b.clicked.connect(lambda checked=False, idx=i: self._switch_tab(idx))
            tb_lay.addWidget(b)
            self._tab_buttons.append(b)
        tb_lay.addStretch(1)
        root.addWidget(tab_bar)

        self.stack = QStackedWidget()
        self.viewer = ViewerTab()
        self.data_tab = DataTab()
        self.train_tab = TrainTab(self)
        self.stack.addWidget(self.viewer)
        self.stack.addWidget(self.data_tab)
        self.stack.addWidget(self.train_tab)
        root.addWidget(self.stack, 1)

        self.relabel = self.data_tab.relabel
        self.viewer.set_capture_for_relabel_callback(self._open_capture_in_relabel)
        self.relabel.set_save_to_train_callback(self._save_relabel_to_train)
        self.relabel.set_ai_assist_callback(self._predict_for_relabel)

        self._switch_tab(0)
        self._bind_shortcuts()

        if self.viewer.model is None:
            QMessageBox.warning(
                self, "Model missing",
                "Default model not found. Use Model → load your best.pt file.",
            )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if sys.platform == "win32":
            QApplication.processEvents()
            hwnd = int(self.winId())
            apply_capture_exclusion_for_hwnd(hwnd, False)

    def _switch_tab(self, index: int) -> None:
        self._current_tab = index
        self.stack.setCurrentIndex(index)
        for i, b in enumerate(self._tab_buttons):
            b.setChecked(i == index)
        if self._active_tab_name() != "Run":
            self.viewer.stop_live_for_tab_switch()

    def _active_tab_name(self) -> str:
        return ["Run", "Data", "Train"][self._current_tab]

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self._on_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._on_right)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._on_space)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._on_ctrl_s)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._on_ctrl_z)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self._on_delete)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._on_escape)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self._on_return)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self._on_return)
        QShortcut(QKeySequence("A"), self, activated=self._on_a)
        QShortcut(QKeySequence("+"), self, activated=self._on_plus)
        QShortcut(QKeySequence("="), self, activated=self._on_plus)
        QShortcut(QKeySequence("-"), self, activated=self._on_minus)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._on_ctrl0)

    def _on_left(self) -> None:
        t = self._active_tab_name()
        if t == "Run":
            self.viewer._prev_image()
        elif t == "Data":
            self.relabel.prev_image()

    def _on_right(self) -> None:
        t = self._active_tab_name()
        if t == "Run":
            self.viewer._next_image()
        elif t == "Data":
            self.relabel.next_image()

    def _on_space(self) -> None:
        if self._active_tab_name() == "Run":
            self.viewer.toggle_autolabel_shortcut()

    def _on_ctrl_s(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel.save_all()

    def _on_ctrl_z(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel.undo_point()

    def _on_delete(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._delete_selected_poi()

    def _on_escape(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._cancel_add()

    def _on_return(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._finish_add_if_ready()

    def _on_a(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._start_add_poi()

    def _on_plus(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._zoom_in()

    def _on_minus(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._zoom_out()

    def _on_ctrl0(self) -> None:
        if self._active_tab_name() == "Data":
            self.relabel._zoom_reset()

    def _open_capture_in_relabel(self, image_path: Path, labels_root: Path) -> None:
        self._switch_tab(1)
        self.data_tab.focus_label_tab()
        self.relabel.open_specific_image(image_path, labels_root=labels_root)

    def _predict_for_relabel(self, bgr_image: np.ndarray) -> list[list[tuple[int, int]]]:
        return self.viewer.predict_for_relabel(bgr_image)

    def _save_relabel_to_train(self, image_path: Path, label_path: Path | None) -> str:
        master_images = DATA_LABELED_DIR / "images"
        master_labels = DATA_LABELED_DIR / "labels"
        master_images.mkdir(parents=True, exist_ok=True)
        master_labels.mkdir(parents=True, exist_ok=True)
        out_image = master_images / image_path.name
        out_label = master_labels / f"{image_path.stem}.txt"
        if image_path.exists():
            shutil.copy2(str(image_path), str(out_image))
        if label_path is not None and label_path.exists():
            shutil.copy2(str(label_path), str(out_label))
            return f"Copied to labeled/{out_image.name}"
        if not out_label.exists():
            out_label.touch()
        return f"Exported negative to labeled/{out_image.name}"

    def load_trained_model(self, path: str) -> None:
        self._switch_tab(0)
        self.viewer.load_trained_model(path)

    def load_result_classifier(self, path: str) -> None:
        self._switch_tab(0)
        self.viewer.load_result_classifier(path)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.viewer.close_cleanup()
        self.data_tab.close_cleanup()
        self.train_tab.close_cleanup()
        event.accept()


def run_app() -> int:
    app = QApplication.instance()
    if app is None:
        try:
            QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        except Exception:
            pass
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(application_stylesheet())
    win = PoiStudioWindow()
    win.show()
    return app.exec()
