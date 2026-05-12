"""Simplified training tab: folder pairs → bundled YOLO dataset → train."""
from __future__ import annotations

import json
import os
import queue
import random
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ultralytics import YOLO

from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    BG_CARD,
    BG_HOVER,
    BG_INPUT,
    BG_PANEL,
    BORDER,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    FG,
    FG_DIM,
    FONT,
    MONO,
    OUTPUTS_DIR,
    TRAIN_BUNDLE_DIR,
    TRAIN_DATA_YAML,
    TRAIN_SOURCES_JSON,
    RED,
    RED_HOVER,
    RUNS_DIR,
    SUPPORTED_EXTENSIONS,
)
from studio.widgets import FlatButton

_VAL_FRACTION = 0.15
_BUNDLE_SEED = 42


def _train_device() -> int | str:
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _collect_pairs_for_source(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    label_by_stem: dict[str, Path] = {}
    for lab_path in labels_dir.rglob("*.txt"):
        if lab_path.is_file():
            label_by_stem[lab_path.stem.lower()] = lab_path
    pairs: list[tuple[Path, Path]] = []
    for img_path in images_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        lab = label_by_stem.get(img_path.stem.lower())
        if lab is not None:
            pairs.append((img_path, lab))
    pairs.sort(key=lambda t: str(t[0]).lower())
    return pairs


def _label_file_has_objects(lab_path: Path) -> bool:
    """True if .txt looks like YOLO lines with enough numbers (has at least one object)."""
    try:
        text = lab_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return False
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            return True
    return False


def _standard_images_dir(root: Path) -> Path | None:
    """Expect layout …/images; return that dir if it exists."""
    d = root / "images"
    return d if d.is_dir() else None


def _standard_labels_dir(root: Path) -> Path | None:
    d = root / "labels"
    return d if d.is_dir() else None


def _collect_positive_pairs_tagged(root: Path | None, source_tag: int) -> list[tuple[Path, Path, int]]:
    """All image+label pairs under root/images + root/labels where label has ≥1 object."""
    if root is None or not root.is_dir():
        return []
    img_d = _standard_images_dir(root)
    lbl_d = _standard_labels_dir(root)
    if img_d is None or lbl_d is None:
        return []
    out: list[tuple[Path, Path, int]] = []
    for img_p, lab_p in _collect_pairs_for_source(img_d, lbl_d):
        if _label_file_has_objects(lab_p):
            out.append((img_p, lab_p, source_tag))
    return out


def _enumerate_negative_images(neg_root: Path | None, labeled_paths: set[str]) -> list[Path]:
    """Images under negatives folder (…/images or root itself), excluding any path in labeled_paths."""
    if neg_root is None or not neg_root.is_dir():
        return []
    img_root = _standard_images_dir(neg_root)
    if img_root is None:
        img_root = neg_root
    found: list[Path] = []
    for p in img_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        key = str(p.resolve())
        if key in labeled_paths:
            continue
        found.append(p)
    found.sort(key=lambda x: str(x).lower())
    return found


def _split_train_val_list(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    if not items:
        return [], []
    rnd = random.Random(seed)
    sh = list(items)
    rnd.shuffle(sh)
    if len(sh) == 1:
        return sh, []
    n_val = max(1, int(round(len(sh) * val_fraction)))
    n_val = min(n_val, len(sh) - 1)
    return sh[n_val:], sh[:n_val]


BundleRow = tuple[Path, Path | None, int]  # image, label or None → empty .txt, source tag


def _reset_bundle_dirs() -> tuple[Path, Path, Path, Path]:
    if TRAIN_BUNDLE_DIR.exists():
        shutil.rmtree(TRAIN_BUNDLE_DIR, ignore_errors=True)
    train_img = TRAIN_BUNDLE_DIR / "images" / "train"
    train_lbl = TRAIN_BUNDLE_DIR / "labels" / "train"
    val_img = TRAIN_BUNDLE_DIR / "images" / "val"
    val_lbl = TRAIN_BUNDLE_DIR / "labels" / "val"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)
    return train_img, train_lbl, val_img, val_lbl


def _copy_bundle_rows(rows: list[BundleRow], img_dir: Path, lbl_dir: Path) -> None:
    for idx, (img_p, lab_p, si) in enumerate(rows):
        stem = f"src{si:02d}_{idx:05d}_{img_p.stem}"[:180]
        dst_i = img_dir / f"{stem}{img_p.suffix}"
        dst_l = lbl_dir / f"{stem}.txt"
        shutil.copy2(img_p, dst_i)
        if lab_p is None:
            dst_l.write_text("", encoding="utf-8")
        else:
            shutil.copy2(lab_p, dst_l)


def _write_data_yaml() -> None:
    root_abs = TRAIN_BUNDLE_DIR.resolve()
    yaml_text = (
        f"path: {root_abs.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: poi\n"
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_YAML.write_text(yaml_text, encoding="utf-8")


def prepare_mixed_train_bundle(
    generated_root: Path | None,
    labeled_poi_root: Path | None,
    negatives_root: Path | None,
    *,
    labeled_fraction: float,
    val_fraction: float = _VAL_FRACTION,
    seed: int = _BUNDLE_SEED,
) -> tuple[Path, int, int, str]:
    """Use all positive (POI) images from generated + labeled roots; add negatives to reach labeled_fraction.

    labeled_fraction: fraction of the **full** dataset that has POI labels (e.g. 0.2 → 20% POI, 80% negatives).
    Returns (yaml_path, n_train_images, n_val_images, summary_text).
    """
    if not (0 < labeled_fraction <= 1.0):
        raise ValueError("Labeled fraction must be between 0 and 100% (e.g. 20 for 20% POI images).")

    tagged: list[tuple[Path, Path, int]] = []
    tagged.extend(_collect_positive_pairs_tagged(generated_root, 0))
    tagged.extend(_collect_positive_pairs_tagged(labeled_poi_root, 1))

    by_res: dict[str, tuple[Path, Path, int]] = {}
    for img_p, lab_p, tag in tagged:
        k = str(img_p.resolve())
        if k not in by_res:
            by_res[k] = (img_p, lab_p, tag)
    labeled_rows: list[tuple[Path, Path, int]] = list(by_res.values())

    if len(labeled_rows) < 1:
        raise ValueError(
            "No images with non-empty POI labels were found under generated / labeled folders "
            "(need …/images + …/labels with matching names).",
        )

    labeled_paths = {str(t[0].resolve()) for t in labeled_rows}
    neg_root_use = negatives_root if (negatives_root and negatives_root.is_dir()) else None
    neg_pool = _enumerate_negative_images(neg_root_use, labeled_paths)

    n_l = len(labeled_rows)
    target_total = int(round(n_l / labeled_fraction))
    n_neg_needed = max(0, target_total - n_l)

    rnd = random.Random(seed + 7)
    neg_pool_shuffled = list(neg_pool)
    rnd.shuffle(neg_pool_shuffled)
    neg_chosen = neg_pool_shuffled[: min(n_neg_needed, len(neg_pool_shuffled))]

    actual_total = n_l + len(neg_chosen)
    actual_labeled_frac = n_l / actual_total if actual_total else 0.0

    train_l, val_l = _split_train_val_list(labeled_rows, val_fraction, seed)
    neg_as_rows: list[tuple[Path, None, int]] = [(p, None, 2) for p in neg_chosen]
    train_n, val_n = _split_train_val_list(neg_as_rows, val_fraction, seed + 1)

    train_bundle: list[BundleRow] = [(a, b, c) for a, b, c in train_l] + [(a, b, c) for a, b, c in train_n]
    val_bundle: list[BundleRow] = [(a, b, c) for a, b, c in val_l] + [(a, b, c) for a, b, c in val_n]

    if len(train_bundle) < 1:
        raise ValueError("Train split ended up empty — add more data.")
    if not val_bundle:
        if len(train_bundle) < 2:
            raise ValueError(
                "After mixing, fewer than 2 images are available for train/val — add more labeled or negative images.",
            )
        val_bundle.append(train_bundle.pop())

    train_img, train_lbl, val_img, val_lbl = _reset_bundle_dirs()
    _copy_bundle_rows(train_bundle, train_img, train_lbl)
    _copy_bundle_rows(val_bundle, val_img, val_lbl)
    _write_data_yaml()

    summary = (
        f"{n_l} POI images (all used), {len(neg_chosen)} negatives "
        f"(wanted {n_neg_needed}, pool had {len(neg_pool)}). "
        f"Mix ≈ {actual_labeled_frac * 100:.1f}% labeled. "
        f"Train {len(train_bundle)} / val {len(val_bundle)}."
    )
    if len(neg_chosen) < n_neg_needed:
        summary += " Not enough negatives for exact ratio — used all available."

    return TRAIN_DATA_YAML.resolve(), len(train_bundle), len(val_bundle), summary


def prepare_train_bundle(
    sources: list[tuple[Path, Path]],
    *,
    val_fraction: float = _VAL_FRACTION,
    seed: int = _BUNDLE_SEED,
) -> tuple[Path, int, int]:
    """Copy matched pairs into a YOLO-OBB layout and write dataset yaml. Returns (yaml_path, n_train, n_val)."""
    all_rows: list[tuple[Path, Path, int]] = []
    for si, (img_d, lbl_d) in enumerate(sources):
        for img_p, lab_p in _collect_pairs_for_source(img_d, lbl_d):
            all_rows.append((img_p, lab_p, si))

    if not all_rows:
        raise ValueError("No image + label pairs found across your folders.")
    if len(all_rows) < 2:
        raise ValueError(
            "Need at least 2 labeled images total (across all folders) to split train / validation."
        )

    rnd = random.Random(seed)
    rnd.shuffle(all_rows)

    n = len(all_rows)
    n_val = max(1, int(round(n * val_fraction)))
    if n <= 3:
        n_val = 1
    n_val = min(n_val, n - 1)

    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    train_img, train_lbl, val_img, val_lbl = _reset_bundle_dirs()
    _copy_bundle_rows(train_rows, train_img, train_lbl)
    _copy_bundle_rows(val_rows, val_img, val_lbl)
    _write_data_yaml()

    return TRAIN_DATA_YAML.resolve(), len(train_rows), len(val_rows)


class TrainingTab(tk.Frame):
    """Train YOLO-OBB: add image+label folders, pick fresh or retrain, run."""

    def __init__(self, parent, main_app):
        super().__init__(parent, bg=BG)
        self.main_app = main_app
        self.training = False
        self.training_thread = None
        self.training_stop_event = threading.Event()
        self._tab_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._tab_ui_poll_id: str | None = None

        self.sources: list[tuple[Path, Path]] = []
        self.retrain_weights: Path | None = None

        self.mix_use_var = tk.BooleanVar(value=False)
        self.mix_generated_root: Path | None = None
        self.mix_labeled_root: Path | None = None
        self.mix_negatives_root: Path | None = None
        self.mix_labeled_pct_var = tk.StringVar(value="20")

        self._build_ui()
        self._schedule_tab_ui_poll()
        self._load_sources_json()
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

        def _sync(_e=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable.bind("<Configure>", _sync)

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(cw_id, width=event.width)
            _sync()

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for w in (canvas, scrollable):
            w.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        pad_x = 24

        hero = tk.Frame(scrollable, bg=BG)
        hero.pack(fill="x", padx=pad_x, pady=(18, 8))
        tk.Frame(hero, bg=ACCENT, height=3).pack(fill="x")
        hi = tk.Frame(hero, bg=BG_PANEL)
        hi.pack(fill="x")
        tk.Label(hi, text="Train", bg=BG_PANEL, fg=FG, font=(FONT, 20, "bold"), anchor="w").pack(
            anchor="w", padx=18, pady=(16, 4)
        )
        tk.Label(
            hi,
            text="Either use the data mix (generated + labeled + negatives) or add manual image/label folder pairs.",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 10),
            anchor="w",
            wraplength=920,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 16))

        mix_card = tk.Frame(scrollable, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        mix_card.pack(fill="x", padx=pad_x, pady=(0, 10))
        tk.Label(
            mix_card,
            text="1. Data mix (optional)",
            bg=BG_CARD,
            fg=FG,
            font=(FONT, 11, "bold"),
        ).pack(anchor="w", padx=18, pady=(12, 4))
        tk.Label(
            mix_card,
            text="Each root should contain subfolders images/ and labels/ (negatives: images/ only). "
            "All POI-labeled images from generated + labeled are kept; negatives are sampled so the "
            "full set matches the Labeled field below (e.g. 20 means about 20% with POI, 80% negatives).",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
            wraplength=920,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 8))
        tk.Checkbutton(
            mix_card,
            text="Use this mix instead of the manual folder list below",
            variable=self.mix_use_var,
            bg=BG_CARD,
            fg=FG,
            activebackground=BG_CARD,
            selectcolor=BG_INPUT,
            font=(FONT, 10, "bold"),
            command=self._save_sources_json,
        ).pack(anchor="w", padx=18, pady=(0, 6))

        def _mix_row(parent, label, browse_cmd, attr_name):
            r = tk.Frame(parent, bg=BG_CARD)
            r.pack(fill="x", padx=18, pady=2)
            tk.Label(r, text=label, bg=BG_CARD, fg=FG, font=(FONT, 9), width=22, anchor="w").pack(side="left")
            FlatButton(r, text="Browse…", command=browse_cmd, bg=BG_CARD, bg_hover=BG_HOVER, width=72, height=26, font_size=8).pack(side="left", padx=(0, 8))
            lbl = tk.Label(r, text="—", bg=BG_CARD, fg=FG_DIM, font=(MONO, 8), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            setattr(self, attr_name, lbl)

        _mix_row(mix_card, "generated", self._browse_mix_generated, "mix_lbl_gen")
        _mix_row(mix_card, "labeled_with_poi", self._browse_mix_labeled, "mix_lbl_lab")
        _mix_row(mix_card, "without_poi_negatives", self._browse_mix_negatives, "mix_lbl_neg")

        pct_fr = tk.Frame(mix_card, bg=BG_CARD)
        pct_fr.pack(fill="x", padx=18, pady=(8, 12))
        tk.Label(pct_fr, text="Labeled % of full set (POI images)", bg=BG_CARD, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
        tk.Entry(
            pct_fr,
            textvariable=self.mix_labeled_pct_var,
            width=5,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG,
            font=(MONO, 10),
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            justify="center",
        ).pack(side="left", padx=(8, 6))
        tk.Label(pct_fr, text="(rest = negatives)", bg=BG_CARD, fg=FG_DIM, font=(FONT, 9)).pack(side="left")

        card = tk.Frame(scrollable, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", padx=pad_x, pady=(0, 10))

        tk.Label(card, text="2. Data — manual folder pairs (if mix above is off)", bg=BG_CARD, fg=FG, font=(FONT, 11, "bold")).pack(
            anchor="w", padx=18, pady=(14, 8)
        )
        row_d = tk.Frame(card, bg=BG_CARD)
        row_d.pack(fill="x", padx=18, pady=(0, 8))
        FlatButton(row_d, text="Add pair…", command=self._add_source_pair, bg=ACCENT, bg_hover=ACCENT_HOVER, width=100, height=30, font_size=9, bold=True).pack(side="left", padx=(0, 8))
        FlatButton(row_d, text="Remove selected", command=self._remove_selected_source, bg=BG_CARD, bg_hover=BG_HOVER, width=120, height=30, font_size=9).pack(side="left")

        list_fr = tk.Frame(card, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        list_fr.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.sources_list = tk.Listbox(
            list_fr,
            bg=BG_INPUT,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            font=(MONO, 8),
            height=8,
            bd=0,
            highlightthickness=0,
        )
        self.sources_list.pack(fill="both", expand=True, padx=6, pady=6)

        tk.Label(card, text="3. Model", bg=BG_CARD, fg=FG, font=(FONT, 11, "bold")).pack(anchor="w", padx=18, pady=(4, 6))
        mode_fr = tk.Frame(card, bg=BG_CARD)
        mode_fr.pack(fill="x", padx=18, pady=(0, 8))
        self.start_mode_var = tk.StringVar(value="fresh")
        tk.Radiobutton(
            mode_fr,
            text="Start fresh (YOLOv8s-OBB pretrained — good default)",
            variable=self.start_mode_var,
            value="fresh",
            bg=BG_CARD,
            fg=FG,
            activebackground=BG_CARD,
            selectcolor=BG_INPUT,
            font=(FONT, 10),
            command=self._on_mode_change,
        ).pack(anchor="w")
        rb2_fr = tk.Frame(card, bg=BG_CARD)
        rb2_fr.pack(fill="x", padx=18, pady=(0, 4))
        tk.Radiobutton(
            rb2_fr,
            text="Retrain / continue from weights:",
            variable=self.start_mode_var,
            value="retrain",
            bg=BG_CARD,
            fg=FG,
            activebackground=BG_CARD,
            selectcolor=BG_INPUT,
            font=(FONT, 10),
            command=self._on_mode_change,
        ).pack(side="left")
        FlatButton(rb2_fr, text="Browse .pt…", command=self._browse_weights, bg=BG_CARD, bg_hover=BG_HOVER, width=88, height=28, font_size=9).pack(side="left", padx=(12, 0))
        self.weights_lbl = tk.Label(rb2_fr, text="(none selected)", bg=BG_CARD, fg=FG_DIM, font=(MONO, 8), anchor="w")
        self.weights_lbl.pack(side="left", fill="x", expand=True, padx=(10, 0))

        tk.Label(card, text="4. Options", bg=BG_CARD, fg=FG, font=(FONT, 11, "bold")).pack(anchor="w", padx=18, pady=(10, 6))
        opt = tk.Frame(card, bg=BG_CARD)
        opt.pack(fill="x", padx=18, pady=(0, 14))

        def _field(parent, label, var, w=8):
            f = tk.Frame(parent, bg=BG_CARD)
            f.pack(side="left", padx=(0, 18))
            tk.Label(f, text=label, bg=BG_CARD, fg=FG_DIM, font=(FONT, 9)).pack(anchor="w")
            tk.Entry(
                f,
                textvariable=var,
                width=w,
                bg=BG_INPUT,
                fg=FG,
                insertbackground=FG,
                font=(MONO, 10),
                bd=0,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=ACCENT,
                justify="center",
            ).pack(anchor="w", pady=(2, 0))

        self.lr_var = tk.StringVar(value=str(DEFAULT_LR))
        self.batch_var = tk.StringVar(value=str(DEFAULT_BATCH))
        self.epochs_var = tk.StringVar(value=str(DEFAULT_EPOCHS))
        self.imgsz_var = tk.StringVar(value=str(DEFAULT_IMGSZ))
        self.patience_var = tk.StringVar(value=str(DEFAULT_PATIENCE))
        _field(opt, "Learning rate", self.lr_var)
        _field(opt, "Batch", self.batch_var)
        _field(opt, "Epochs", self.epochs_var)
        _field(opt, "Image size", self.imgsz_var)
        _field(opt, "Patience", self.patience_var)

        run = tk.Frame(scrollable, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        run.pack(fill="x", padx=pad_x, pady=(0, 12))
        tk.Label(run, text="5. Run", bg=BG_CARD, fg=FG, font=(FONT, 11, "bold")).pack(anchor="w", padx=18, pady=(12, 8))
        row_r = tk.Frame(run, bg=BG_CARD)
        row_r.pack(fill="x", padx=18, pady=(0, 8))
        self.start_btn = FlatButton(
            row_r,
            text="Start training",
            command=self._start_training,
            bg=ACCENT,
            bg_hover=ACCENT_HOVER,
            width=130,
            height=36,
            font_size=10,
            bold=True,
        )
        self.start_btn.pack(side="left", padx=(0, 10))
        self.stop_btn = FlatButton(
            row_r,
            text="Stop",
            command=self._stop_training,
            bg=RED,
            bg_hover=RED_HOVER,
            width=80,
            height=36,
            font_size=10,
            bold=True,
        )
        self.stop_btn.pack(side="left", padx=(0, 14))
        self.stop_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER)
        self.progress = ttk.Progressbar(
            row_r,
            maximum=100,
            length=200,
            mode="indeterminate",
            style="PoiTrain.Horizontal.TProgressbar",
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))

        strip = tk.Frame(run, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        strip.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(strip, text="Status", bg=BG_INPUT, fg=ACCENT, font=(FONT, 8, "bold")).pack(side="left", padx=(12, 8), pady=10)
        self.train_status = tk.Label(
            strip,
            text="Turn on data mix or add manual pairs, then start.",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.train_status.pack(side="left", fill="x", expand=True, pady=10, padx=(0, 12))

        foot = tk.Frame(scrollable, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        foot.pack(fill="x", padx=pad_x, pady=(0, 24))
        tk.Label(foot, text="Checkpoints", bg=BG_CARD, fg=FG, font=(FONT, 11, "bold")).pack(anchor="w", padx=18, pady=(12, 6))
        row_f = tk.Frame(foot, bg=BG_CARD)
        row_f.pack(fill="x", padx=18, pady=(0, 10))
        FlatButton(row_f, text="Open runs folder", command=self._open_runs, bg=BG_CARD, bg_hover=BG_HOVER, width=130, height=30, font_size=9).pack(side="left")
        self.models_list = tk.Listbox(
            foot,
            bg=BG_PANEL,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            font=(MONO, 8),
            height=6,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            activestyle="none",
        )
        self.models_list.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        self._refresh_sources_listbox()
        self._refresh_models_list()

    def _on_mode_change(self) -> None:
        pass

    def _mix_initial_dir(self) -> str:
        return str(Path.home())

    def _browse_mix_generated(self) -> None:
        d = filedialog.askdirectory(
            title="generated folder (expects images/ + labels/)",
            parent=self.winfo_toplevel(),
            initialdir=self._mix_initial_dir(),
        )
        if not d:
            return
        self.mix_generated_root = Path(d).resolve()
        self.mix_lbl_gen.configure(text=str(self.mix_generated_root))
        self._save_sources_json()

    def _browse_mix_labeled(self) -> None:
        d = filedialog.askdirectory(
            title="Labeled folder (expects images/ + labels/)",
            parent=self.winfo_toplevel(),
            initialdir=self._mix_initial_dir(),
        )
        if not d:
            return
        self.mix_labeled_root = Path(d).resolve()
        self.mix_lbl_lab.configure(text=str(self.mix_labeled_root))
        self._save_sources_json()

    def _browse_mix_negatives(self) -> None:
        d = filedialog.askdirectory(
            title="Negatives folder (expects images/ with no POI)",
            parent=self.winfo_toplevel(),
            initialdir=self._mix_initial_dir(),
        )
        if not d:
            return
        self.mix_negatives_root = Path(d).resolve()
        self.mix_lbl_neg.configure(text=str(self.mix_negatives_root))
        self._save_sources_json()

    def _sources_json_path(self) -> Path:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        return TRAIN_SOURCES_JSON

    def _save_sources_json(self) -> None:
        try:
            payload = {
                "pairs": [[str(a), str(b)] for a, b in self.sources],
                "mix": {
                    "use": self.mix_use_var.get(),
                    "generated": str(self.mix_generated_root) if self.mix_generated_root else "",
                    "labeled_with_poi": str(self.mix_labeled_root) if self.mix_labeled_root else "",
                    "negatives": str(self.mix_negatives_root) if self.mix_negatives_root else "",
                    "labeled_pct": self.mix_labeled_pct_var.get(),
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
            self._refresh_sources_listbox()
            mx = data.get("mix") or {}
            self.mix_use_var.set(bool(mx.get("use")))
            g, l, n = mx.get("generated") or "", mx.get("labeled_with_poi") or "", mx.get("negatives") or ""
            self.mix_generated_root = Path(g).resolve() if g else None
            self.mix_labeled_root = Path(l).resolve() if l else None
            self.mix_negatives_root = Path(n).resolve() if n else None
            if mx.get("labeled_pct"):
                self.mix_labeled_pct_var.set(str(mx["labeled_pct"]))
            if hasattr(self, "mix_lbl_gen"):
                self.mix_lbl_gen.configure(text=str(self.mix_generated_root) if self.mix_generated_root else "—")
                self.mix_lbl_lab.configure(text=str(self.mix_labeled_root) if self.mix_labeled_root else "—")
                self.mix_lbl_neg.configure(text=str(self.mix_negatives_root) if self.mix_negatives_root else "—")
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _refresh_sources_listbox(self) -> None:
        self.sources_list.delete(0, "end")
        for i, (img_d, lbl_d) in enumerate(self.sources):
            self.sources_list.insert("end", f"{i + 1}.  Images: {img_d}  |  Labels: {lbl_d}")

    def _add_source_pair(self) -> None:
        top = self.winfo_toplevel()
        img = filedialog.askdirectory(title="Images folder (this source)", parent=top)
        if not img:
            return
        lbl = filedialog.askdirectory(title="Labels folder (.txt for this source)", parent=top)
        if not lbl:
            return
        self.sources.append((Path(img).resolve(), Path(lbl).resolve()))
        self._refresh_sources_listbox()
        self._save_sources_json()

    def _remove_selected_source(self) -> None:
        sel = self.sources_list.curselection()
        if not sel:
            messagebox.showinfo("Remove", "Select a row in the list first.")
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.sources):
            del self.sources[idx]
            self._refresh_sources_listbox()
            self._save_sources_json()

    def _browse_weights(self) -> None:
        top = self.winfo_toplevel()
        f = filedialog.askopenfilename(
            title="Weights to continue from (.pt)",
            parent=top,
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        if not f:
            return
        self.retrain_weights = Path(f).resolve()
        self.start_mode_var.set("retrain")
        self.weights_lbl.configure(text=str(self.retrain_weights))

    def _model_source(self) -> str:
        if self.start_mode_var.get() == "retrain":
            if self.retrain_weights is None or not self.retrain_weights.is_file():
                raise ValueError("Choose a .pt file for retrain / continue.")
            return str(self.retrain_weights)
        return "yolov8s-obb.pt"

    def _model_source_ok(self, src: str) -> bool:
        low = src.lower()
        if "yolov" in low and src.endswith(".pt") and not Path(src).is_absolute():
            return True
        return Path(src).expanduser().resolve().is_file()

    def _start_training(self) -> None:
        if self.training:
            messagebox.showwarning("Busy", "Training is already running.")
            return

        use_mix = self.mix_use_var.get()
        labeled_frac = 0.0
        if use_mix:
            if not self.mix_generated_root and not self.mix_labeled_root:
                messagebox.showwarning("Data", "Pick at least one of generated or labeled.")
                return
            try:
                pct = float(self.mix_labeled_pct_var.get().strip())
                labeled_frac = pct / 100.0
            except ValueError:
                messagebox.showerror("Data", "Enter a number for Labeled % (e.g. 20 for 20% POI, 80% negatives).")
                return
            if not (0 < labeled_frac <= 1.0):
                messagebox.showerror("Data", "Labeled % must be between 0 and 100 (exclusive of 0).")
                return
            if labeled_frac < 1.0 and (not self.mix_negatives_root or not self.mix_negatives_root.is_dir()):
                messagebox.showwarning(
                    "Data",
                    "Pick a negatives folder (images/ only), or set Labeled % to 100 if you only want POI images.",
                )
                return
        elif not self.sources:
            messagebox.showwarning("Data", "Turn on data mix or add at least one manual images + labels folder pair.")
            return

        try:
            lr = float(self.lr_var.get())
            batch = int(self.batch_var.get())
            epochs = int(self.epochs_var.get())
            imgsz = int(self.imgsz_var.get())
            patience = int(self.patience_var.get())
        except ValueError:
            messagebox.showerror("Options", "Enter valid numbers for LR, batch, epochs, image size, patience.")
            return

        try:
            model_src = self._model_source()
        except ValueError as e:
            messagebox.showwarning("Model", str(e))
            return

        if not self._model_source_ok(model_src):
            messagebox.showerror("Model not found", f"Cannot load: {model_src}")
            return

        sources_copy = list(self.sources)
        mix_summary_preview = ""
        if use_mix:
            mix_summary_preview = (
                f"Mix mode: generated={self.mix_generated_root or '—'}, "
                f"labeled={self.mix_labeled_root or '—'}, "
                f"negatives={self.mix_negatives_root}\n"
                f"Labeled fraction ≈ {labeled_frac * 100:.0f}% of full set."
            )
        if not messagebox.askyesno(
            "Start training",
            f"Model: {model_src}\n"
            + (mix_summary_preview + "\n" if mix_summary_preview else f"Manual sources: {len(sources_copy)} pair(s)\n")
            + f"Train/val split from that set: ~{int(100 * (1 - _VAL_FRACTION))}% / ~{int(100 * _VAL_FRACTION)}%\n"
            f"Epochs={epochs}, batch={batch}, imgsz={imgsz}\n\n"
            f"Continue?",
        ):
            return

        self.training = True
        self.training_stop_event.clear()
        self.start_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER)
        self.stop_btn.configure_colors(bg=RED, bg_hover=RED_HOVER)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.train_status.configure(text="Preparing dataset…")
        self._save_sources_json()

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
                    self._tab_ui_queue.put(("train_status", f"Training… ({n_tr} train / {n_val} val images)"))
                model = YOLO(model_src)
                device = _train_device()
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
        self.train_status.configure(text="Stop requested (YOLO may finish current epoch)…")
        self._training_complete(False, "Stopped by user.")

    def _training_complete(self, success: bool, message: str) -> None:
        self.training = False
        try:
            self.progress.stop()
        except tk.TclError:
            pass
        self.start_btn.configure_colors(bg=ACCENT, bg_hover=ACCENT_HOVER)
        self.stop_btn.configure_colors(bg=BG_CARD, bg_hover=BG_HOVER)
        self.train_status.configure(text=message)
        self._refresh_models_list()
        if success:
            try:
                candidates = list((RUNS_DIR / "poi_train").rglob("best.pt")) if (RUNS_DIR / "poi_train").exists() else []
                candidates += list(RUNS_DIR.rglob("best.pt"))
                if candidates:
                    last_model = max(candidates, key=lambda p: p.stat().st_mtime)
                    if last_model.exists() and messagebox.askyesno("Load model", "Open the Viewer and load this best.pt?"):
                        self.main_app.load_trained_model(str(last_model))
                else:
                    messagebox.showinfo("Done", message)
            except Exception as e:
                messagebox.showinfo("Done", f"{message}\n({e})")
        else:
            messagebox.showwarning("Training", message)

    def _open_runs(self) -> None:
        proj = RUNS_DIR / "poi_train"
        if proj.exists():
            os.startfile(str(proj))
        elif RUNS_DIR.exists():
            os.startfile(str(RUNS_DIR))
        else:
            messagebox.showinfo("Runs", "No runs folder yet.")

    def _refresh_models_list(self) -> None:
        self.models_list.delete(0, "end")
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
                        self.models_list.insert("end", s)
        except OSError:
            pass

    def _on_close(self) -> None:
        if self._tab_ui_poll_id is not None:
            try:
                self.main_app.root.after_cancel(self._tab_ui_poll_id)
            except tk.TclError:
                pass
            self._tab_ui_poll_id = None
        self.training_stop_event.set()
        if self.training_thread is not None and self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)
