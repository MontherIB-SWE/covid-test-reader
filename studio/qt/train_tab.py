"""Train tab (PySide6)."""
from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
)
from ultralytics import YOLO

from studio.config import (
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    FG_DIM,
    OUTPUTS_DIR,
    RUNS_DIR,
    TRAIN_SOURCES_JSON,
)
from studio.train_ops import VAL_FRACTION, prepare_mixed_train_bundle, prepare_train_bundle, train_device
from studio.qt.widgets import make_button


class TrainTab(QWidget):
    def __init__(self, main_window: QWidget) -> None:
        super().__init__()
        self._main = main_window
        self.training = False
        self.training_thread: threading.Thread | None = None
        self.training_stop_event = threading.Event()
        self._tab_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._process_tab_ui_queue)
        self._poll.start(200)

        self.sources: list[tuple[Path, Path]] = []
        self.retrain_weights: Path | None = None
        self.mix_generated_root: Path | None = None
        self.mix_labeled_root: Path | None = None
        self.mix_negatives_root: Path | None = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        v = QVBoxLayout(self)
        v.addWidget(scroll)
        root = QVBoxLayout(inner)

        root.addWidget(QLabel("<h2>Train</h2>"))
        root.addWidget(QLabel("Use data mix or manual folder pairs."))

        mix = QGroupBox("1. Data mix (optional)")
        ml = QVBoxLayout(mix)
        self.mix_use = QCheckBox("Use this mix instead of manual folder list below")
        self.mix_use.toggled.connect(self._save_sources_json)
        ml.addWidget(self.mix_use)
        self.mix_lbl_gen = QLabel("—")
        self.mix_lbl_lab = QLabel("—")
        self.mix_lbl_neg = QLabel("—")
        for lbl, txt, slot in [
            (self.mix_lbl_gen, "generated", self._browse_mix_generated),
            (self.mix_lbl_lab, "labeled_with_poi", self._browse_mix_labeled),
            (self.mix_lbl_neg, "negatives", self._browse_mix_negatives),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(txt))
            row.addWidget(make_button("Browse…", slot, tooltip=f"Select {txt} folder"))
            row.addWidget(lbl, 1)
            ml.addLayout(row)
        pct_row = QHBoxLayout()
        pct_row.addWidget(QLabel("Labeled % of full set"))
        self.mix_labeled_pct = QLineEdit("20")
        self.mix_labeled_pct.setFixedWidth(48)
        pct_row.addWidget(self.mix_labeled_pct)
        ml.addLayout(pct_row)
        root.addWidget(mix)

        man = QGroupBox("2. Manual folder pairs")
        man_l = QVBoxLayout(man)
        br = QHBoxLayout()
        br.addWidget(make_button("Add pair…", self._add_source_pair, style="primary",
                                 tooltip="Pick images folder then labels folder"))
        br.addWidget(make_button("Remove selected", self._remove_selected_source,
                                 tooltip="Remove highlighted pair from list"))
        man_l.addLayout(br)
        self.sources_list = QListWidget()
        self.sources_list.setMinimumHeight(120)
        man_l.addWidget(self.sources_list)
        root.addWidget(man)

        mod = QGroupBox("3. Model")
        mod_l = QVBoxLayout(mod)
        self._mode_group = QButtonGroup(self)
        self._rb_fresh = QRadioButton("Start fresh (yolov8s-obb.pt)")
        self._rb_retrain = QRadioButton("Retrain from weights:")
        self._rb_fresh.setChecked(True)
        self._mode_group.addButton(self._rb_fresh)
        self._mode_group.addButton(self._rb_retrain)
        mod_l.addWidget(self._rb_fresh)
        rw = QHBoxLayout()
        rw.addWidget(self._rb_retrain)
        rw.addWidget(make_button("Browse .pt…", self._browse_weights, tooltip="Weights for continue training"))
        self.weights_lbl = QLabel("(none)")
        self.weights_lbl.setStyleSheet(f"color:{FG_DIM};")
        rw.addWidget(self.weights_lbl, 1)
        mod_l.addLayout(rw)
        root.addWidget(mod)

        opt = QGroupBox("4. Options")
        og = QGridLayout(opt)
        self.lr = QLineEdit(str(DEFAULT_LR))
        self.batch = QLineEdit(str(DEFAULT_BATCH))
        self.epochs = QLineEdit(str(DEFAULT_EPOCHS))
        self.imgsz = QLineEdit(str(DEFAULT_IMGSZ))
        self.patience = QLineEdit(str(DEFAULT_PATIENCE))
        for i, (name, w) in enumerate([
            ("LR", self.lr), ("Batch", self.batch), ("Epochs", self.epochs),
            ("Image size", self.imgsz), ("Patience", self.patience),
        ]):
            og.addWidget(QLabel(name), 0, i)
            w.setFixedWidth(72)
            og.addWidget(w, 1, i)
        root.addWidget(opt)

        run = QGroupBox("5. Run")
        rl = QVBoxLayout(run)
        rr = QHBoxLayout()
        self.start_btn = make_button("Start training", self._start_training, style="primary",
                                     min_width=140, tooltip="Build dataset bundle and run YOLO train")
        self.stop_btn = make_button("Stop", self._stop_training, style="danger",
                                    min_width=90, tooltip="Request training stop")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        rr.addWidget(self.start_btn)
        rr.addWidget(self.stop_btn)
        rr.addWidget(self.progress, 1)
        rl.addLayout(rr)
        self.train_status = QLabel("Add data sources, then start.")
        self.train_status.setWordWrap(True)
        rl.addWidget(self.train_status)
        root.addWidget(run)

        foot = QGroupBox("Checkpoints")
        fl = QVBoxLayout(foot)
        fl.addWidget(make_button("Open runs folder", self._open_runs,
                                 tooltip="Open runs/poi_train in Explorer"))
        self.models_list = QListWidget()
        self.models_list.setMinimumHeight(100)
        fl.addWidget(self.models_list)
        root.addWidget(foot)
        root.addStretch(1)

        self._load_sources_json()
        self._refresh_sources_listbox()
        self._refresh_models_list()

    def _process_tab_ui_queue(self) -> None:
        while True:
            try:
                item = self._tab_ui_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "train_status":
                self.train_status.setText(item[1])
            elif item[0] == "train_complete":
                self._training_complete(item[1], item[2])

    def _sources_json_path(self) -> Path:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        return TRAIN_SOURCES_JSON

    def _save_sources_json(self) -> None:
        try:
            payload = {
                "pairs": [[str(a), str(b)] for a, b in self.sources],
                "mix": {
                    "use": self.mix_use.isChecked(),
                    "generated": str(self.mix_generated_root) if self.mix_generated_root else "",
                    "labeled_with_poi": str(self.mix_labeled_root) if self.mix_labeled_root else "",
                    "negatives": str(self.mix_negatives_root) if self.mix_negatives_root else "",
                    "labeled_pct": self.mix_labeled_pct.text(),
                },
            }
            self._sources_json_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_sources_json(self) -> None:
        p = self._sources_json_path()
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            pairs = data.get("pairs") or []
            self.sources = []
            for row in pairs:
                if len(row) >= 2 and row[0] and row[1]:
                    self.sources.append((Path(row[0]).resolve(), Path(row[1]).resolve()))
            mx = data.get("mix") or {}
            self.mix_use.setChecked(bool(mx.get("use")))
            g, l, n = mx.get("generated") or "", mx.get("labeled_with_poi") or "", mx.get("negatives") or ""
            self.mix_generated_root = Path(g).resolve() if g else None
            self.mix_labeled_root = Path(l).resolve() if l else None
            self.mix_negatives_root = Path(n).resolve() if n else None
            if mx.get("labeled_pct"):
                self.mix_labeled_pct.setText(str(mx["labeled_pct"]))
            self.mix_lbl_gen.setText(str(self.mix_generated_root) if self.mix_generated_root else "—")
            self.mix_lbl_lab.setText(str(self.mix_labeled_root) if self.mix_labeled_root else "—")
            self.mix_lbl_neg.setText(str(self.mix_negatives_root) if self.mix_negatives_root else "—")
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _browse_mix_generated(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "generated folder")
        if d:
            self.mix_generated_root = Path(d).resolve()
            self.mix_lbl_gen.setText(str(self.mix_generated_root))
            self._save_sources_json()

    def _browse_mix_labeled(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "labeled folder")
        if d:
            self.mix_labeled_root = Path(d).resolve()
            self.mix_lbl_lab.setText(str(self.mix_labeled_root))
            self._save_sources_json()

    def _browse_mix_negatives(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "negatives folder")
        if d:
            self.mix_negatives_root = Path(d).resolve()
            self.mix_lbl_neg.setText(str(self.mix_negatives_root))
            self._save_sources_json()

    def _refresh_sources_listbox(self) -> None:
        self.sources_list.clear()
        for i, (img_d, lbl_d) in enumerate(self.sources):
            self.sources_list.addItem(f"{i + 1}. Images: {img_d} | Labels: {lbl_d}")

    def _add_source_pair(self) -> None:
        img = QFileDialog.getExistingDirectory(self, "Images folder")
        if not img:
            return
        lbl = QFileDialog.getExistingDirectory(self, "Labels folder")
        if not lbl:
            return
        self.sources.append((Path(img).resolve(), Path(lbl).resolve()))
        self._refresh_sources_listbox()
        self._save_sources_json()

    def _remove_selected_source(self) -> None:
        row = self.sources_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Remove", "Select a row first.")
            return
        if 0 <= row < len(self.sources):
            del self.sources[row]
            self._refresh_sources_listbox()
            self._save_sources_json()

    def _browse_weights(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Weights", "", "PyTorch (*.pt)")
        if f:
            self.retrain_weights = Path(f).resolve()
            self._rb_retrain.setChecked(True)
            self.weights_lbl.setText(str(self.retrain_weights))

    def _model_source(self) -> str:
        if self._rb_retrain.isChecked():
            if self.retrain_weights is None or not self.retrain_weights.is_file():
                raise ValueError("Choose a .pt file for retrain.")
            return str(self.retrain_weights)
        return "yolov8s-obb.pt"

    def _model_source_ok(self, src: str) -> bool:
        low = src.lower()
        if "yolov" in low and src.endswith(".pt") and not Path(src).is_absolute():
            return True
        return Path(src).expanduser().resolve().is_file()

    def _start_training(self) -> None:
        if self.training:
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return
        use_mix = self.mix_use.isChecked()
        labeled_frac = 0.0
        if use_mix:
            if not self.mix_generated_root and not self.mix_labeled_root:
                QMessageBox.warning(self, "Data", "Pick at least one of generated or labeled.")
                return
            try:
                pct = float(self.mix_labeled_pct.text().strip())
                labeled_frac = pct / 100.0
            except ValueError:
                QMessageBox.critical(self, "Data", "Enter Labeled % as a number (e.g. 20).")
                return
            if not (0 < labeled_frac <= 1.0):
                QMessageBox.critical(self, "Data", "Labeled % must be between 0 and 100.")
                return
            if labeled_frac < 1.0 and (not self.mix_negatives_root or not self.mix_negatives_root.is_dir()):
                QMessageBox.warning(self, "Data", "Pick a negatives folder or set Labeled % to 100.")
                return
        elif not self.sources:
            QMessageBox.warning(self, "Data", "Turn on data mix or add manual pairs.")
            return
        try:
            lr = float(self.lr.text())
            batch = int(self.batch.text())
            epochs = int(self.epochs.text())
            imgsz = int(self.imgsz.text())
            patience = int(self.patience.text())
        except ValueError:
            QMessageBox.critical(self, "Options", "Enter valid numbers for LR, batch, epochs, imgsz, patience.")
            return
        try:
            model_src = self._model_source()
        except ValueError as e:
            QMessageBox.warning(self, "Model", str(e))
            return
        if not self._model_source_ok(model_src):
            QMessageBox.critical(self, "Model not found", f"Cannot load: {model_src}")
            return
        mix_preview = ""
        if use_mix:
            mix_preview = (
                f"Mix: gen={self.mix_generated_root or '—'}, lab={self.mix_labeled_root or '—'}, "
                f"neg={self.mix_negatives_root}\nLabeled ≈ {labeled_frac * 100:.0f}%.\n"
            )
        if QMessageBox.question(
            self, "Start training",
            f"Model: {model_src}\n{mix_preview}"
            f"Epochs={epochs}, batch={batch}, imgsz={imgsz}\n\nContinue?",
        ) != QMessageBox.StandardButton.Yes:
            return

        self.training = True
        self.training_stop_event.clear()
        self.progress.setRange(0, 0)
        self.train_status.setText("Preparing dataset…")
        self._save_sources_json()
        sources_copy = list(self.sources)

        def _worker() -> None:
            try:
                if use_mix:
                    yaml_path, n_tr, n_val, mix_sum = prepare_mixed_train_bundle(
                        self.mix_generated_root,
                        self.mix_labeled_root,
                        self.mix_negatives_root,
                        labeled_fraction=labeled_frac,
                    )
                    self._tab_ui_queue.put(
                        ("train_status", f"Training… ({n_tr} train / {n_val} val). {mix_sum}"),
                    )
                else:
                    yaml_path, n_tr, n_val = prepare_train_bundle(sources_copy)
                    self._tab_ui_queue.put(("train_status", f"Training… ({n_tr} train / {n_val} val)"))
                model = YOLO(model_src)
                device = train_device()
                model.train(
                    data=str(yaml_path),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    lr0=lr,
                    lrf=0.01,
                    patience=patience,
                    device=device,
                    project=str(RUNS_DIR / "poi_train"),
                    name="exp",
                    exist_ok=True,
                    resume=False,
                    amp=True,
                    close_mosaic=10,
                    hsv_h=0.02,
                    hsv_s=0.8,
                    hsv_v=0.5,
                    degrees=15,
                    translate=0.2,
                    scale=0.9,
                    shear=5.0,
                    fliplr=0.5,
                    flipud=0.1,
                    mosaic=0.5,
                    auto_augment="randaugment",
                    dropout=0.15,
                    cls=0.5,
                    box=7.5,
                    angle=1.0,
                    warmup_epochs=3.0,
                    warmup_bias_lr=0.1,
                )
                self._tab_ui_queue.put(("train_complete", True, "Training finished."))
            except Exception as e:
                self._tab_ui_queue.put(("train_complete", False, f"Error: {e}"))

        self.training_thread = threading.Thread(target=_worker, daemon=True)
        self.training_thread.start()

    def _stop_training(self) -> None:
        if not self.training:
            return
        self.training_stop_event.set()
        self.train_status.setText("Stop requested…")
        self._training_complete(False, "Stopped by user.")

    def _training_complete(self, success: bool, message: str) -> None:
        self.training = False
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.train_status.setText(message)
        self._refresh_models_list()
        if success:
            try:
                candidates = list((RUNS_DIR / "poi_train").rglob("best.pt")) if (RUNS_DIR / "poi_train").exists() else []
                candidates += list(RUNS_DIR.rglob("best.pt"))
                if candidates:
                    last_model = max(candidates, key=lambda p: p.stat().st_mtime)
                    if last_model.exists() and QMessageBox.question(
                        self, "Load model", "Switch to Run tab and load this best.pt?",
                    ) == QMessageBox.StandardButton.Yes:
                        mw = self._main
                        if hasattr(mw, "load_trained_model"):
                            mw.load_trained_model(str(last_model))
                else:
                    QMessageBox.information(self, "Done", message)
            except Exception as e:
                QMessageBox.information(self, "Done", f"{message}\n({e})")
        else:
            QMessageBox.warning(self, "Training", message)

    def _open_runs(self) -> None:
        proj = RUNS_DIR / "poi_train"
        if proj.exists():
            os.startfile(str(proj))
        elif RUNS_DIR.exists():
            os.startfile(str(RUNS_DIR))
        else:
            QMessageBox.information(self, "Runs", "No runs folder yet.")

    def _refresh_models_list(self) -> None:
        self.models_list.clear()
        try:
            roots = [RUNS_DIR / "poi_train", RUNS_DIR]
            seen: set[str] = set()
            for root in roots:
                if not root.exists():
                    continue
                for p in root.rglob("best.pt"):
                    s = str(p.resolve())
                    if s not in seen:
                        seen.add(s)
                        self.models_list.addItem(s)
        except OSError:
            pass

    def close_cleanup(self) -> None:
        self._poll.stop()
        self.training_stop_event.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)
