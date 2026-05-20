"""Data tab: relabel + dataset generator (PySide6)."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import studio.pipeline as pipeline
from studio.config import DATASET_PATHS_JSON, FG_DIM, SUPPORTED_EXTENSIONS
from studio.pipeline import extract_assets
from studio.qt.relabel_widget import RelabelWidget
from studio.qt.result_label_widget import ResultLabelWidget
from studio.qt.widgets import make_button


class DataTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.backgrounds_dir: Path | None = None
        self.output_dir: Path | None = None
        self._extracting = False
        self._generating = False
        self._tab_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._process_tab_ui_queue)
        self._poll_timer.start(120)

        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.relabel = RelabelWidget()
        self.tabs.addTab(self.relabel, "Label")
        dataset = QWidget()
        self._build_dataset_panel(dataset)
        self.tabs.addTab(dataset, "Dataset")
        self.result_label = ResultLabelWidget()
        self.tabs.addTab(self.result_label, "Results")
        lay.addWidget(self.tabs)
        self.relabel.label_saved.connect(self.result_label.mark_needs_regen)
        self._try_restore_paths()

    def focus_label_tab(self) -> None:
        self.tabs.setCurrentIndex(0)

    def _process_tab_ui_queue(self) -> None:
        while True:
            try:
                item = self._tab_ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "pipeline":
                self.pipeline_status.setText(item[1])
            elif kind == "extract_done":
                self._extracting = False
            elif kind == "generate_done":
                self._generating = False

    def _build_dataset_panel(self, parent: QWidget) -> None:
        v = QVBoxLayout(parent)
        hero = QGroupBox("Synthetic images")
        hl = QVBoxLayout(hero)
        hl.addWidget(QLabel(
            "Images + labels define POIs. Backgrounds are negatives. Output → output/images and output/labels."
        ))
        v.addWidget(hero)

        card = QGroupBox("Folders")
        g = QGridLayout(card)
        r = 0
        g.addWidget(
            make_button("Choose images folder…", self._pick_images_dir, style="primary",
                        tooltip="Folder with source POI images", min_width=200),
            r, 0,
        )
        g.addWidget(
            make_button("Choose labels folder…", self._pick_labels_dir,
                        tooltip="Matching YOLO .txt labels", min_width=200),
            r, 1,
        )
        r += 1
        g.addWidget(
            make_button("Choose output folder…", self._pick_output_dir, style="primary",
                        tooltip="Synthetic images + labels output", min_width=200),
            r, 0,
        )
        g.addWidget(
            make_button("Choose backgrounds…", self._pick_backgrounds_dir,
                        tooltip="Images without POI (paste targets)", min_width=200),
            r, 1,
        )
        r += 1
        self.path_images_lbl = QLabel("Images: —")
        self.path_labels_lbl = QLabel("Labels: —")
        self.path_output_lbl = QLabel("Output: —")
        self.path_backgrounds_lbl = QLabel("Backgrounds: —")
        for lb in (self.path_images_lbl, self.path_labels_lbl, self.path_output_lbl, self.path_backgrounds_lbl):
            lb.setWordWrap(True)
            lb.setObjectName("pathLabel")
        g.addWidget(self.path_images_lbl, r, 0, 1, 2)
        r += 1
        g.addWidget(self.path_labels_lbl, r, 0, 1, 2)
        r += 1
        g.addWidget(self.path_output_lbl, r, 0, 1, 2)
        r += 1
        g.addWidget(self.path_backgrounds_lbl, r, 0, 1, 2)
        v.addWidget(card)

        row = QHBoxLayout()
        row.addWidget(
            make_button("Extract POI crops", self._run_extract, style="primary", min_width=160,
                tooltip="Crop POIs from images+labels into cache/assets"),
        )
        row.addWidget(QLabel("Step 1 — builds assets under output/.poi_studio_cache/assets"))
        v.addLayout(row)

        gen = QHBoxLayout()
        gen.addWidget(QLabel("Count:"))
        self.synth_count = QLineEdit("50")
        self.synth_count.setFixedWidth(72)
        self.synth_count.setToolTip("Number of synthetic composites to generate")
        gen.addWidget(self.synth_count)
        gen.addWidget(
            make_button("Generate composites", self._run_generate, style="primary", min_width=180,
                tooltip="Step 2 — paste assets onto background images"),
        )
        gen.addWidget(QLabel("Writes output/images + output/labels"))
        gen.addStretch(1)
        v.addLayout(gen)

        st = QHBoxLayout()
        st.addWidget(QLabel("<b>Status</b>"))
        self.pipeline_status = QLabel("Choose folders → Extract → Generate.")
        self.pipeline_status.setWordWrap(True)
        self.pipeline_status.setObjectName("statusLabel")
        st.addWidget(self.pipeline_status, 1)
        v.addLayout(st)
        v.addStretch(1)

    def _paths_json_path(self) -> Path:
        return DATASET_PATHS_JSON.resolve()

    def _save_paths_json(self) -> None:
        try:
            payload = {
                "images_dir": str(self.images_dir) if self.images_dir else "",
                "labels_dir": str(self.labels_dir) if self.labels_dir else "",
                "backgrounds_dir": str(self.backgrounds_dir) if self.backgrounds_dir else "",
                "output_dir": str(self.output_dir) if self.output_dir else "",
            }
            self._paths_json_path().parent.mkdir(parents=True, exist_ok=True)
            self._paths_json_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _try_restore_paths(self) -> None:
        j = self._paths_json_path()
        if not j.is_file():
            return
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
            img = data.get("images_dir") or ""
            lab = data.get("labels_dir") or ""
            self.images_dir = Path(img).resolve() if img else None
            self.labels_dir = Path(lab).resolve() if lab else None
            out = data.get("output_dir") or ""
            if out:
                self.output_dir = Path(out).resolve()
            elif data.get("work_dir"):
                self.output_dir = Path(data["work_dir"]).resolve() / "data_master" / "synthetic_output"
            bg = data.get("backgrounds_dir") or ""
            if bg:
                self.backgrounds_dir = Path(bg).resolve()
            elif data.get("work_dir"):
                legacy_bg = Path(data["work_dir"]).resolve() / "data_master" / "without_poi_negatives" / "images"
                if legacy_bg.is_dir():
                    self.backgrounds_dir = legacy_bg
            if self.images_dir:
                self.path_images_lbl.setText(f"Images: {self.images_dir}")
            if self.labels_dir:
                self.path_labels_lbl.setText(f"Labels: {self.labels_dir}")
            if self.output_dir:
                self.path_output_lbl.setText(f"Output: {self.output_dir}")
            if self.backgrounds_dir:
                self.path_backgrounds_lbl.setText(f"Backgrounds: {self.backgrounds_dir}")
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _pick_images_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Images folder")
        if not d:
            return
        self.images_dir = Path(d).resolve()
        self.path_images_lbl.setText(f"Images: {self.images_dir}")
        self._save_paths_json()

    def _pick_labels_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Labels folder")
        if not d:
            return
        self.labels_dir = Path(d).resolve()
        self.path_labels_lbl.setText(f"Labels: {self.labels_dir}")
        self._save_paths_json()

    def _pick_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder")
        if not d:
            return
        self.output_dir = Path(d).resolve()
        self.path_output_lbl.setText(f"Output: {self.output_dir}")
        self._save_paths_json()

    def _pick_backgrounds_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Backgrounds folder")
        if not d:
            return
        self.backgrounds_dir = Path(d).resolve()
        self.path_backgrounds_lbl.setText(f"Backgrounds: {self.backgrounds_dir}")
        self._save_paths_json()

    def _count_background_images(self, folder: Path) -> int:
        n = 0
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                n += 1
        return n

    def _cache_assets_dir(self) -> Path:
        assert self.output_dir is not None
        return self.output_dir / ".poi_studio_cache" / "assets"

    def _count_cached_assets(self) -> int:
        if self.output_dir is None:
            return 0
        cache = self._cache_assets_dir()
        if not cache.is_dir():
            return 0
        ok_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sum(1 for p in cache.rglob("*") if p.is_file() and p.suffix.lower() in ok_ext)

    def _pipeline_busy(self) -> bool:
        return self._extracting or self._generating

    def _collect_pairs(self) -> list[tuple[Path, Path]]:
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

    def _run_extract(self) -> None:
        if self._pipeline_busy():
            return
        if not self.images_dir or not self.labels_dir or not self.output_dir:
            QMessageBox.warning(self, "Folders", "Choose Images, Labels, and Output folders.")
            return
        pairs = self._collect_pairs()
        if not pairs:
            QMessageBox.information(self, "No pairs", "No matching image + .txt pairs found.")
            return
        images_dir = self.images_dir
        labels_dir = self.labels_dir
        cache_assets = self._cache_assets_dir()
        self._extracting = True
        self.pipeline_status.setText(f"Extracting… ({len(pairs)} pair(s))")

        def _worker() -> None:
            try:
                cache_assets.mkdir(parents=True, exist_ok=True)
                stats = extract_assets(
                    images_dirs=[images_dir],
                    labels_dirs=[labels_dir],
                    output_dir=cache_assets,
                )
                extracted = stats.get("extracted", 0) if isinstance(stats, dict) else 0
                if extracted == 0:
                    self._tab_ui_queue.put(("pipeline", "❌ Extract produced no crops."))
                    return
                self._tab_ui_queue.put(("pipeline", f"✅ Extract done — {extracted} asset(s). Run Generate next."))
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Extract error: {e}"))
            finally:
                self._tab_ui_queue.put(("extract_done",))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_generate(self) -> None:
        if self._pipeline_busy():
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Folders", "Choose an Output folder first.")
            return
        if self._count_cached_assets() == 0:
            QMessageBox.warning(self, "Nothing to paste", "Run Extract first.")
            return
        if not self.backgrounds_dir or not self.backgrounds_dir.is_dir():
            QMessageBox.warning(self, "Backgrounds", "Choose a backgrounds folder.")
            return
        if self._count_background_images(self.backgrounds_dir) == 0:
            QMessageBox.warning(self, "No backgrounds", "No images found in backgrounds folder.")
            return
        try:
            n = int(self.synth_count.text().strip())
            if n < 1:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Invalid count", "Enter a positive integer.")
            return
        backgrounds_dir = self.backgrounds_dir
        output_dir = self.output_dir
        cache_assets = self._cache_assets_dir()
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"
        self._generating = True
        self.pipeline_status.setText(f"Generating… {n} image(s)")

        def _worker() -> None:
            try:
                out_img.mkdir(parents=True, exist_ok=True)
                out_lbl.mkdir(parents=True, exist_ok=True)
                pipeline.generate(
                    num_images=n,
                    on_saved=None,
                    bg_dir=backgrounds_dir,
                    assets_dir=cache_assets,
                    out_img_dir=out_img,
                    out_lbl_dir=out_lbl,
                )
                self._tab_ui_queue.put(("pipeline", f"✅ Done — {n} synthetic image(s) → {out_img}"))
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Generate error: {e}"))
            finally:
                self._tab_ui_queue.put(("generate_done",))

        threading.Thread(target=_worker, daemon=True).start()

    def close_cleanup(self) -> None:
        self._poll_timer.stop()
