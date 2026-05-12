"""Unified POI Studio shell (Run + Data + Train)."""
from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path

import numpy as np
from tkinter import messagebox

from studio.config import (
    ACCENT,
    BG,
    BG_CARD,
    BG_PANEL,
    DATA_LABELED_DIR,
    FG,
    FONT,
)
from studio.data_tab import DataManagementTab
from studio.train_tab import TrainingTab
from studio.viewer import PoiDesktopViewer

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
        def _left(_):
            tab = self._active_tab_name()
            if tab == "Run":
                self.viewer._prev_image()
            elif tab == "Data":
                self.relabel.prev_image()

        def _right(_):
            tab = self._active_tab_name()
            if tab == "Run":
                self.viewer._next_image()
            elif tab == "Data":
                self.relabel.next_image()

        self.root.bind("<Left>", _left)
        self.root.bind("<Right>", _right)
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
