"""Results sub-tab: prepare warped crops and classify (positive/negative/invalid)."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from studio.config import (
    ACCENT,
    BG_CARD,
    BG_PANEL,
    BORDER,
    DATA_LABELED_DIR,
    FG_DIM,
    MONO,
    ORANGE,
    RED,
    RESULT_CLS_AUG_PER_STEM,
    RESULT_CLS_CROPS_DIR,
    RESULT_CLS_MANIFEST,
    SUPPORTED_EXTENSIONS,
)
from studio.qt.images import numpy_bgr_to_qpixmap
from studio.qt.widgets import make_button
from studio.result_cls_ops import (
    RESULT_CLASSES,
    SOURCE_LABELED,
    ManifestEntry,
    export_crops_for_folder,
    generate_augmented_crops,
    read_manifest,
    upsert_manifest_entry,
    load_result_classifier,
)

_CLASS_COLORS = {"positive": "#10b981", "negative": "#ef4444", "invalid": "#f59e0b"}
_CLASS_KEYS = {"positive": "1", "negative": "2", "invalid": "3"}


class ResultLabelWidget(QWidget):
    """Classification labeling for antigen test result crops."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images_dir: Path | None = None
        self._labels_dir: Path | None = None
        self._manifest: dict[str, str] = read_manifest()
        self._stems: list[str] = []
        self._crop_status: dict[str, str] = {}
        self._current_idx: int = -1
        self._classifier = None
        self._ui_queue: queue.Queue[tuple] = queue.Queue()

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._process_queue)
        self._poll.start(100)

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("<b style='color:#00a86b;'>Results</b> — antigen read labeling"))
        toolbar.addStretch(1)
        toolbar.addWidget(make_button("Open folder…", self._open_folder, style="primary"))
        toolbar.addWidget(make_button("Regen all crops", self._regen_crops, tooltip="Re-export all 448×448 crops"))
        aug_row = QHBoxLayout()
        aug_row.addWidget(QLabel("Aug / stem:"))
        self._aug_per_stem = QLineEdit(str(RESULT_CLS_AUG_PER_STEM))
        self._aug_per_stem.setFixedWidth(40)
        self._aug_per_stem.setToolTip("Number of augmented copies per classified image")
        aug_row.addWidget(self._aug_per_stem)
        aug_row.addWidget(make_button(
            "Generate augmented",
            self._generate_augmented,
            style="primary",
            tooltip="Create variants under data/result_cls/augmented/ (classifier training only)",
        ))
        aug_row.addStretch(1)
        toolbar.addLayout(aug_row)
        toolbar.addWidget(make_button("Build bundle", self._build_bundle, style="primary",
                                      tooltip="Original crops + augmented → outputs/result_cls_bundle"))
        root.addLayout(toolbar)
        hint = QLabel("Live saves from Run tab appear here after refresh or when you open this tab.")
        hint.setStyleSheet(f"color:{FG_DIM}; font-size:9pt;")
        root.addWidget(hint)

        # Main area with Tab Widget
        self._tab_widget = QTabWidget()
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Tab 1: Labeler
        labeler_widget = QWidget()
        labeler_layout = QHBoxLayout(labeler_widget)
        labeler_layout.setContentsMargins(0, 4, 0, 4)

        # Left: queue list + filter
        left = QVBoxLayout()
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems([
            "All", "Live", "Unlabeled", "Positive", "Negative", "Invalid", "No POI",
        ])
        self._filter_combo.currentIndexChanged.connect(self._refresh_queue_list)
        filter_row.addWidget(self._filter_combo)
        filter_row.addStretch(1)
        left.addLayout(filter_row)

        self._queue_list = QListWidget()
        self._queue_list.setMaximumWidth(280)
        self._queue_list.currentRowChanged.connect(self._on_queue_select)
        left.addWidget(self._queue_list, 1)

        nav_row = QHBoxLayout()
        nav_row.addWidget(make_button("◀", self._prev, style="compact"))
        nav_row.addWidget(make_button("▶", self._next, style="compact"))
        nav_row.addWidget(make_button("Next unlabeled (U)", self._jump_unlabeled, style="compact"))
        left.addLayout(nav_row)

        self._counts_lbl = QLabel("")
        self._counts_lbl.setStyleSheet(f"color:{FG_DIM}; font-size:9pt;")
        left.addWidget(self._counts_lbl)

        # Right: preview + classify buttons
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Training crop (448×448)</b> — what ResNet sees"))
        self._crop_preview = QLabel()
        self._crop_preview.setMinimumSize(280, 280)
        self._crop_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_preview.setObjectName("editorCanvas")
        right.addWidget(self._crop_preview, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color:{FG_DIM};")
        right.addWidget(self._status_lbl)

        # Classify buttons
        btn_row = QHBoxLayout()
        self._btn_pos = make_button("POSITIVE (1)", lambda: self._classify("positive"), style="primary")
        self._btn_neg = make_button("NEGATIVE (2)", lambda: self._classify("negative"), style="danger")
        self._btn_inv = make_button("INVALID (3)", lambda: self._classify("invalid"), style="warning")
        self._btn_pos.setObjectName("resultPosButton")
        self._btn_neg.setObjectName("resultNegButton")
        self._btn_inv.setObjectName("resultInvButton")
        btn_row.addWidget(self._btn_pos)
        btn_row.addWidget(self._btn_neg)
        btn_row.addWidget(self._btn_inv)
        right.addLayout(btn_row)

        # AI suggest row
        self._suggest_lbl = QLabel("")
        self._suggest_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:bold;")
        self._btn_accept = make_button("Accept", self._accept_suggestion, style="compact")
        self._btn_accept.setVisible(False)
        suggest_row = QHBoxLayout()
        suggest_row.addWidget(self._suggest_lbl, 1)
        suggest_row.addWidget(self._btn_accept)
        right.addLayout(suggest_row)

        labeler_layout.addLayout(left)
        labeler_layout.addLayout(right, 1)

        self._tab_widget.addTab(labeler_widget, "Labeler")

        # Tab 2: Gallery Review
        self._gallery_widget = QWidget()
        self._build_gallery_ui()
        self._tab_widget.addTab(self._gallery_widget, "Gallery Review")

        root.addWidget(self._tab_widget, 1)

        # Status bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(18)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("1"), self).activated.connect(lambda: self._classify("positive"))
        QShortcut(QKeySequence("2"), self).activated.connect(lambda: self._classify("negative"))
        QShortcut(QKeySequence("3"), self).activated.connect(lambda: self._classify("invalid"))
        QShortcut(QKeySequence("U"), self).activated.connect(self._jump_unlabeled)

    # ─── Folder + manifest ────────────────────────────────────────────────────

    def reload_manifest(self) -> None:
        self._manifest = read_manifest()
        self._refresh_queue_list()
        self._update_counts()

    def _open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose images folder", str(DATA_LABELED_DIR))
        if not d:
            return
        images_dir = Path(d)
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            labels_dir = images_dir / ".." / "labels"
            labels_dir = labels_dir.resolve()
        if not labels_dir.is_dir():
            labels_dir_pick = QFileDialog.getExistingDirectory(self, "Choose labels folder", str(images_dir.parent))
            if not labels_dir_pick:
                return
            labels_dir = Path(labels_dir_pick)
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        self._manifest = read_manifest()
        self._scan_and_prepare()

    def _scan_and_prepare(self) -> None:
        if not self._images_dir or not self._images_dir.is_dir():
            return
        image_files = sorted(
            [f for f in self._images_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        )
        self._stems = [f.stem for f in image_files]
        self._regen_crops()

    def _regen_crops(self) -> None:
        if not self._images_dir or not self._labels_dir:
            QMessageBox.warning(self, "No folder", "Open an images folder first.")
            return
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._stems))
        self._progress.setValue(0)

        images_dir = self._images_dir
        labels_dir = self._labels_dir

        def _worker():
            results = export_crops_for_folder(
                images_dir, labels_dir,
                progress_cb=lambda i, t, s: self._ui_queue.put(("crop_progress", i, t, s)),
            )
            self._ui_queue.put(("crop_done", results))

        threading.Thread(target=_worker, daemon=True).start()

    def _process_queue(self) -> None:
        while True:
            try:
                item = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "crop_progress":
                _, i, total, stem = item
                self._progress.setRange(0, total)
                self._progress.setValue(i)
            elif kind == "crop_done":
                self._crop_status = item[1]
                self._progress.setVisible(False)
                self._refresh_queue_list()
                self._update_counts()
                if self._stems:
                    self._select_index(0)
            elif kind == "aug_progress":
                _, i, total, stem = item
                self._progress.setRange(0, total)
                self._progress.setValue(i)
            elif kind == "aug_done":
                self._progress.setVisible(False)
                ok, payload = item[1], item[2]
                if ok:
                    QMessageBox.information(
                        self,
                        "Augmented",
                        f"Wrote {payload['images_written']} images "
                        f"({payload['per_stem']} per stem, {payload['stems_processed']} stems).\n"
                        f"{payload['augmented_dir']}",
                    )
                else:
                    QMessageBox.critical(self, "Augment failed", payload)

    # ─── Queue list ───────────────────────────────────────────────────────────

    def _refresh_queue_list(self) -> None:
        self._queue_list.blockSignals(True)
        self._queue_list.clear()
        filt = self._filter_combo.currentText().lower()
        for stem in self._stems:
            cls = self._manifest.get(stem)
            status = self._crop_status.get(stem, "")
            if filt == "live" and not stem.startswith("live_"):
                continue
            if filt == "unlabeled" and cls is not None:
                continue
            if filt == "positive" and cls != "positive":
                continue
            if filt == "negative" and cls != "negative":
                continue
            if filt == "invalid" and cls != "invalid":
                continue
            if filt == "no poi" and status != "no_label":
                continue
            label_text = cls or "unlabeled"
            item = QListWidgetItem(f"{stem}  [{label_text}]")
            if cls:
                item.setForeground(self._queue_list.palette().text().color())
            else:
                item.setForeground(self._queue_list.palette().placeholderText().color())
            item.setData(Qt.ItemDataRole.UserRole, stem)
            self._queue_list.addItem(item)
        self._queue_list.blockSignals(False)
        self._update_counts()

    def _on_queue_select(self, row: int) -> None:
        if row < 0:
            return
        item = self._queue_list.item(row)
        if item is None:
            return
        stem = item.data(Qt.ItemDataRole.UserRole)
        idx = self._stems.index(stem) if stem in self._stems else -1
        if idx >= 0:
            self._select_index(idx)

    def _select_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._stems):
            return
        self._current_idx = idx
        stem = self._stems[idx]
        crop_path = RESULT_CLS_CROPS_DIR / f"{stem}.jpg"
        status = self._crop_status.get(stem, "")

        if crop_path.is_file():
            bgr = cv2.imread(str(crop_path))
            if bgr is not None:
                pm = numpy_bgr_to_qpixmap(bgr)
                self._crop_preview.setPixmap(pm.scaled(
                    280, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))
            else:
                self._crop_preview.setText("Failed to read crop")
        elif status == "no_label":
            self._crop_preview.setText("No POI label — add on Label tab")
        elif status == "warp_failed":
            self._crop_preview.setText("Warp failed — fix corners on Label tab")
        else:
            self._crop_preview.setText("No crop — run Regen")

        cls = self._manifest.get(stem)
        if cls:
            self._status_lbl.setText(f"Current: <b style='color:{_CLASS_COLORS[cls]}'>{cls.upper()}</b>")
            self._crop_preview.setStyleSheet(
                f"background:{BG_CARD}; border:3px solid {_CLASS_COLORS[cls]};"
            )
        else:
            self._status_lbl.setText("Unlabeled")
            self._crop_preview.setStyleSheet(f"background:{BG_CARD}; border:2px solid {BORDER};")

        enabled = status == "ready" or crop_path.is_file()
        self._btn_pos.setEnabled(enabled)
        self._btn_neg.setEnabled(enabled)
        self._btn_inv.setEnabled(True)

        self._run_ai_suggest(stem)

    # ─── Classification ───────────────────────────────────────────────────────

    def _classify(self, cls: str) -> None:
        if self._current_idx < 0 or self._current_idx >= len(self._stems):
            return
        stem = self._stems[self._current_idx]
        session_id = self._images_dir.name if self._images_dir else "legacy"
        self._manifest[stem] = cls
        upsert_manifest_entry(
            ManifestEntry(stem=stem, cls=cls, source=SOURCE_LABELED, session_id=session_id),
        )
        self._refresh_queue_list()
        self._update_counts()
        self._advance_after_classify()

    def _advance_after_classify(self) -> None:
        """Auto-advance to next unlabeled."""
        for i in range(self._current_idx + 1, len(self._stems)):
            if self._stems[i] not in self._manifest:
                self._select_index(i)
                return
        for i in range(0, self._current_idx):
            if self._stems[i] not in self._manifest:
                self._select_index(i)
                return
        if self._current_idx + 1 < len(self._stems):
            self._select_index(self._current_idx + 1)

    def _accept_suggestion(self) -> None:
        text = self._suggest_lbl.text()
        for cls in RESULT_CLASSES:
            if cls.upper() in text:
                self._classify(cls)
                return

    # ─── AI suggest ───────────────────────────────────────────────────────────

    def _run_ai_suggest(self, stem: str) -> None:
        self._suggest_lbl.setText("")
        self._btn_accept.setVisible(False)
        if self._classifier is None:
            self._classifier = load_result_classifier()
        if self._classifier is None:
            return
        crop_path = RESULT_CLS_CROPS_DIR / f"{stem}.jpg"
        if not crop_path.is_file():
            return
        bgr = cv2.imread(str(crop_path))
        if bgr is None:
            return
        cls_name, conf = self._classifier.predict(bgr)
        self._suggest_lbl.setText(f"AI suggest: {cls_name.upper()} ({conf:.2f})")
        self._btn_accept.setVisible(True)

    # ─── Navigation ───────────────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._current_idx > 0:
            self._select_index(self._current_idx - 1)

    def _next(self) -> None:
        if self._current_idx < len(self._stems) - 1:
            self._select_index(self._current_idx + 1)

    def _jump_unlabeled(self) -> None:
        start = self._current_idx + 1 if self._current_idx >= 0 else 0
        for i in range(start, len(self._stems)):
            if self._stems[i] not in self._manifest:
                self._select_index(i)
                return
        for i in range(0, start):
            if self._stems[i] not in self._manifest:
                self._select_index(i)
                return

    # ─── Stats ────────────────────────────────────────────────────────────────

    def _update_counts(self) -> None:
        total = len(self._stems)
        labeled = sum(1 for s in self._stems if s in self._manifest)
        pos = sum(1 for s in self._stems if self._manifest.get(s) == "positive")
        neg = sum(1 for s in self._stems if self._manifest.get(s) == "negative")
        inv = sum(1 for s in self._stems if self._manifest.get(s) == "invalid")
        unlab = total - labeled
        self._counts_lbl.setText(
            f"{labeled}/{total} labeled | {unlab} unlabeled | "
            f"pos:{pos} neg:{neg} inv:{inv}"
        )

    def _generate_augmented(self) -> None:
        labeled = {s: c for s, c in self._manifest.items() if c in RESULT_CLASSES}
        if not labeled:
            QMessageBox.warning(self, "No labels", "Classify some images first (positive / negative / invalid).")
            return
        try:
            per_stem = int(self._aug_per_stem.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Invalid", "Enter a number for Aug / stem (e.g. 12).")
            return
        if per_stem < 1:
            QMessageBox.critical(self, "Invalid", "Aug / stem must be at least 1.")
            return

        self._progress.setVisible(True)
        self._progress.setRange(0, len(labeled))
        self._progress.setValue(0)
        manifest_copy = dict(labeled)

        def _worker():
            try:
                summary = generate_augmented_crops(
                    manifest_copy,
                    per_stem=per_stem,
                    progress_cb=lambda i, t, s: self._ui_queue.put(("aug_progress", i, t, s)),
                )
                self._ui_queue.put(("aug_done", True, summary))
            except Exception as e:
                self._ui_queue.put(("aug_done", False, str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    # ─── Bundle ───────────────────────────────────────────────────────────────

    def _build_bundle(self) -> None:
        from studio.result_cls_ops import build_result_cls_bundle
        labeled_count = sum(1 for c in self._manifest.values() if c in RESULT_CLASSES)
        if labeled_count < 3:
            QMessageBox.warning(self, "Not enough", "Label at least a few images before building.")
            return
        try:
            bundle_dir, n_train, n_val, stats = build_result_cls_bundle(self._manifest)
            QMessageBox.information(
                self, "Bundle built",
                f"Train: {stats['n_train_total']} ({stats['n_orig_train']} orig + {stats['n_aug_train']} aug)\n"
                f"Val: {stats['n_val_total']} ({stats['n_orig_val']} orig + {stats['n_aug_val']} aug)\n"
                f"{bundle_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ─── Public API for sync ──────────────────────────────────────────────────

    def mark_needs_regen(self, stem: str) -> None:
        """Called externally when POI corners change for a stem."""
        self._crop_status[stem] = "needs_regen"

    def reload_manifest(self) -> None:
        self._manifest = read_manifest()
        self._refresh_queue_list()
        self._update_counts()

    def _build_gallery_ui(self) -> None:
        layout = QVBoxLayout(self._gallery_widget)
        layout.setContentsMargins(4, 4, 4, 4)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Filter:"))
        self._gallery_filter = QComboBox()
        self._gallery_filter.addItems(["All", "Unlabeled", "Positive", "Negative", "Invalid", "No POI"])
        self._gallery_filter.currentIndexChanged.connect(self._refresh_gallery)
        top_bar.addWidget(self._gallery_filter)

        self._gallery_stats = QLabel("0 items")
        self._gallery_stats.setStyleSheet(f"color:{FG_DIM}; font-size:9pt; margin-left:8px;")
        top_bar.addWidget(self._gallery_stats)
        top_bar.addStretch(1)
        top_bar.addWidget(make_button("Refresh", self._refresh_gallery, style="compact"))
        layout.addLayout(top_bar)

        self._gallery_scroll = QScrollArea()
        self._gallery_scroll.setWidgetResizable(True)
        self._gallery_scroll.setObjectName("cropsScrollArea")

        self._gallery_container = QWidget()
        self._gallery_grid = QGridLayout(self._gallery_container)
        self._gallery_grid.setSpacing(12)
        self._gallery_grid.setContentsMargins(8, 8, 8, 8)

        self._gallery_scroll.setWidget(self._gallery_container)
        layout.addWidget(self._gallery_scroll, 1)

    def _create_gallery_card(self, stem: str, cls: str) -> QWidget:
        card = QWidget()
        card.setObjectName("editorCanvas")
        card.setFixedSize(140, 190)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # Double-click to jump to Labeler tab and select
        def dblClick(event, s=stem):
            self._select_image_by_stem(s)
            self._tab_widget.setCurrentIndex(0)

        card.mouseDoubleClickEvent = dblClick

        # Image preview
        lbl_img = QLabel()
        lbl_img.setFixedSize(128, 128)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_img.setObjectName("leftImageLabel")

        # Color-coded borders matching the category structure
        cls_colors = {"positive": "#10b981", "negative": "#ef4444", "invalid": "#f59e0b", "unlabeled": "#25293d"}
        color = cls_colors.get(cls.lower(), "#25293d")
        lbl_img.setStyleSheet(f"border: 2px solid {color}; border-radius: 4px;")

        crop_path = RESULT_CLS_CROPS_DIR / f"{stem}.jpg"
        status = self._crop_status.get(stem, "")

        if crop_path.is_file():
            bgr = cv2.imread(str(crop_path))
            if bgr is not None:
                from studio.qt.images import numpy_bgr_to_qpixmap
                pm = numpy_bgr_to_qpixmap(bgr)
                lbl_img.setPixmap(pm.scaled(
                    120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))
            else:
                lbl_img.setText("Read Error")
        elif status == "no_label":
            lbl_img.setText("No POI")
        elif status == "warp_failed":
            lbl_img.setText("Warp Failed")
        else:
            lbl_img.setText("No Crop")

        lay.addWidget(lbl_img)

        # Stem name
        lbl_stem = QLabel(stem)
        lbl_stem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_stem.setStyleSheet("font-size: 8pt; font-weight: bold;")
        lbl_stem.setToolTip(stem)
        lay.addWidget(lbl_stem)

        # Class selector dropdown including "Unlabeled"
        cb = QComboBox()
        cb.addItems(["Unlabeled", "Positive", "Negative", "Invalid"])
        cb.setCurrentText(cls.capitalize())
        cb.blockSignals(True)
        cb.setCurrentText(cls.capitalize())
        cb.blockSignals(False)

        def on_change(text, s=stem):
            new_cls = text.lower()
            if new_cls == "unlabeled":
                if s in self._manifest:
                    del self._manifest[s]
            else:
                self._manifest[s] = new_cls
            from studio.result_cls_ops import write_manifest
            write_manifest(self._manifest)
            self._refresh_queue_list()
            self._update_counts()
            self._refresh_gallery()

        cb.currentTextChanged.connect(on_change)
        lay.addWidget(cb)

        return card

    def _refresh_gallery(self) -> None:
        # Clear existing layout items
        while self._gallery_grid.count():
            item = self._gallery_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        filt = self._gallery_filter.currentText().lower()

        # Combine folder stems and manifest stems (preserving order and uniqueness)
        all_stems = list(self._stems)
        manifest_stems = sorted(self._manifest.keys())
        for stem in manifest_stems:
            if stem not in all_stems:
                all_stems.append(stem)

        matched_stems = []
        for stem in all_stems:
            cls = self._manifest.get(stem)
            status = self._crop_status.get(stem, "")

            # If not in self._stems and not classified, it shouldn't exist in the manifest context
            if not cls and stem not in self._stems:
                continue

            if filt == "live" and not stem.startswith("live_"):
                continue
            if filt == "unlabeled" and cls is not None:
                continue
            if filt == "positive" and cls != "positive":
                continue
            if filt == "negative" and cls != "negative":
                continue
            if filt == "invalid" and cls != "invalid":
                continue
            if filt == "no poi" and status != "no_label":
                continue

            matched_stems.append((stem, cls or "unlabeled"))

        if not matched_stems:
            lbl = QLabel("No images match this filter.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._gallery_grid.addWidget(lbl, 0, 0)
            self._gallery_stats.setText("0 items")
            return

        cols = 5
        for idx, (stem, cls) in enumerate(matched_stems):
            r = idx // cols
            c = idx % cols
            card = self._create_gallery_card(stem, cls)
            self._gallery_grid.addWidget(card, r, c)
        self._gallery_stats.setText(f"{len(matched_stems)} items")

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._refresh_gallery()

    def _select_image_by_stem(self, stem: str) -> None:
        self._filter_combo.blockSignals(True)
        self._filter_combo.setCurrentText("All")
        self._filter_combo.blockSignals(False)
        self._refresh_queue_list()

        for idx in range(self._queue_list.count()):
            item = self._queue_list.item(idx)
            if item.data(Qt.ItemDataRole.UserRole) == stem:
                self._queue_list.setCurrentRow(idx)
                break
