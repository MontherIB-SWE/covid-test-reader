"""Data tab: labeling host + simple synthetic dataset generator."""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import studio.pipeline as pipeline
from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    BG_CARD,
    BG_HOVER,
    BG_INPUT,
    BG_PANEL,
    BORDER,
    FG,
    FG_DIM,
    FONT,
    MONO,
    SUPPORTED_EXTENSIONS,
    DATASET_PATHS_JSON,
)
from studio.pipeline import extract_assets
from studio.relabel import RelabelCornersTool
from studio.widgets import FlatButton


class DataManagementTab(tk.Frame):
    """Label surface + Dataset: source folders → synthetic images + YOLO labels in one output folder."""

    def __init__(self, parent, main_app):
        super().__init__(parent, bg=BG)
        self.main_app = main_app
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.backgrounds_dir: Path | None = None
        self.output_dir: Path | None = None
        self._extracting = False
        self._generating = False
        self._tab_ui_queue: queue.Queue[tuple] = queue.Queue()
        self._tab_ui_poll_id: str | None = None

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        label_host = tk.Frame(self.notebook, bg=BG)
        dataset_host = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(label_host, text="Label")
        self.notebook.add(dataset_host, text="Dataset")

        self.relabel = RelabelCornersTool(label_host, embedded=True, bind_shortcuts=False)
        self._build_dataset_panel(dataset_host)
        self._try_restore_paths()
        self._schedule_tab_ui_poll()
        self.pack(fill=tk.BOTH, expand=True)

    def focus_label_tab(self) -> None:
        try:
            self.notebook.select(0)
        except tk.TclError:
            pass

    def _schedule_tab_ui_poll(self) -> None:
        self._process_tab_ui_queue()
        try:
            self._tab_ui_poll_id = self.main_app.root.after(120, self._schedule_tab_ui_poll)
        except tk.TclError:
            self._tab_ui_poll_id = None

    def _process_tab_ui_queue(self) -> None:
        while True:
            try:
                item = self._tab_ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "pipeline":
                self.pipeline_status.configure(text=item[1])
            elif kind == "extract_done":
                self._extracting = False
            elif kind == "generate_done":
                self._generating = False

    def _build_dataset_panel(self, parent: tk.Frame) -> None:
        pad_x = 24
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True)

        hero = tk.Frame(outer, bg=BG_PANEL)
        hero.pack(fill=tk.X, padx=pad_x, pady=(20, 12))
        tk.Frame(hero, bg=ACCENT, height=3).pack(fill=tk.X)
        inner = tk.Frame(hero, bg=BG_PANEL)
        inner.pack(fill=tk.X, padx=18, pady=16)
        tk.Label(
            inner,
            text="Synthetic images",
            bg=BG_PANEL,
            fg=FG,
            font=(FONT, 20, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            inner,
            text=(
                "Images + labels define POIs (cropped for pasting). "
                "Backgrounds must be separate: frames with no POI (negatives / empty screens). "
                "Outputs go under your output folder."
            ),
            bg=BG_PANEL,
            fg=FG_DIM,
            font=(FONT, 10),
            anchor="w",
            wraplength=920,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        card = tk.Frame(
            outer,
            bg=BG_CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.pack(fill=tk.X, padx=pad_x, pady=(0, 12))

        flow = tk.Frame(card, bg=BG_CARD)
        flow.pack(fill=tk.X, padx=18, pady=(18, 10))
        FlatButton(
            flow,
            text="Images folder",
            command=self._pick_images_dir,
            bg=ACCENT,
            bg_hover=ACCENT_HOVER,
            width=124,
            height=32,
            font_size=9,
            bold=True,
        ).pack(side="left", padx=(0, 8))
        FlatButton(
            flow,
            text="Labels folder",
            command=self._pick_labels_dir,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=124,
            height=32,
            font_size=9,
        ).pack(side="left", padx=(0, 8))
        FlatButton(
            flow,
            text="Output folder",
            command=self._pick_output_dir,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=124,
            height=32,
            font_size=9,
            bold=True,
        ).pack(side="left", padx=(0, 8))

        flow2 = tk.Frame(card, bg=BG_CARD)
        flow2.pack(fill=tk.X, padx=18, pady=(0, 10))
        FlatButton(
            flow2,
            text="Backgrounds (no POI)",
            command=self._pick_backgrounds_dir,
            bg=ACCENT,
            bg_hover=ACCENT_HOVER,
            width=168,
            height=32,
            font_size=9,
            bold=True,
        ).pack(side="left", padx=(0, 8))

        paths = tk.Frame(card, bg=BG_CARD)
        paths.pack(fill=tk.X, padx=18, pady=(0, 14))
        self.path_images_lbl = tk.Label(
            paths,
            text="Images: —",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_images_lbl.pack(fill=tk.X)
        self.path_labels_lbl = tk.Label(
            paths,
            text="Labels: —",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_labels_lbl.pack(fill=tk.X)
        self.path_output_lbl = tk.Label(
            paths,
            text="Output: —",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_output_lbl.pack(fill=tk.X)
        self.path_backgrounds_lbl = tk.Label(
            paths,
            text="Backgrounds (no POI): —",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.path_backgrounds_lbl.pack(fill=tk.X)

        tk.Label(
            paths,
            text="Writes: output/images/ (synth_*.jpg) and output/labels/ (matching .txt). "
            "Intermediate crops are kept in output/.poi_studio_cache/assets (safe to delete after).",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
            wraplength=900,
            justify="left",
        ).pack(fill=tk.X, pady=(12, 0))

        extract_row = tk.Frame(card, bg=BG_CARD)
        extract_row.pack(fill=tk.X, padx=18, pady=(0, 10))
        FlatButton(
            extract_row,
            text="Extract",
            command=self._run_extract,
            bg=ACCENT,
            bg_hover=ACCENT_HOVER,
            width=110,
            height=34,
            font_size=10,
            bold=True,
        ).pack(side="left")
        tk.Label(
            extract_row,
            text="Crop POIs from Images + Labels → output/.poi_studio_cache/assets",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
        ).pack(side="left", padx=(12, 0))

        gen_row = tk.Frame(card, bg=BG_CARD)
        gen_row.pack(fill=tk.X, padx=18, pady=(0, 18))
        tk.Label(gen_row, text="How many to generate", bg=BG_CARD, fg=FG, font=(FONT, 10)).pack(
            side="left",
            padx=(0, 10),
        )
        self.synth_count_var = tk.StringVar(value="50")
        tk.Entry(
            gen_row,
            textvariable=self.synth_count_var,
            width=8,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG,
            font=(MONO, 10),
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            justify="center",
        ).pack(side="left", padx=(0, 16))
        FlatButton(
            gen_row,
            text="Generate",
            command=self._run_generate,
            bg=BG_CARD,
            bg_hover=BG_HOVER,
            width=110,
            height=34,
            font_size=10,
            bold=True,
        ).pack(side="left")
        tk.Label(
            gen_row,
            text="Paste assets onto Backgrounds → output/images + labels",
            bg=BG_CARD,
            fg=FG_DIM,
            font=(FONT, 9),
            anchor="w",
        ).pack(side="left", padx=(12, 0))

        status_bar = tk.Frame(card, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
        status_bar.pack(fill=tk.X, padx=18, pady=(0, 18))
        tk.Label(
            status_bar,
            text="Status",
            bg=BG_INPUT,
            fg=ACCENT,
            font=(FONT, 8, "bold"),
        ).pack(side="left", padx=(12, 8), pady=10)
        self.pipeline_status = tk.Label(
            status_bar,
            text="Choose folders → Extract (once or after label changes) → Generate.",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=(MONO, 9),
            anchor="w",
        )
        self.pipeline_status.pack(side="left", fill=tk.X, expand=True, pady=10, padx=(0, 12))

    # ── Paths / persistence ───────────────────────────────────────────────

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
                # Legacy save: working folder only — old synthetics lived here
                self.output_dir = Path(data["work_dir"]).resolve() / "data_master" / "synthetic_output"
            bg = data.get("backgrounds_dir") or ""
            if bg:
                self.backgrounds_dir = Path(bg).resolve()
            elif data.get("work_dir"):
                legacy_bg = Path(data["work_dir"]).resolve() / "data_master" / "without_poi_negatives" / "images"
                if legacy_bg.is_dir():
                    self.backgrounds_dir = legacy_bg
            if self.images_dir:
                self.path_images_lbl.configure(text=f"Images: {self.images_dir}")
            if self.labels_dir:
                self.path_labels_lbl.configure(text=f"Labels: {self.labels_dir}")
            if self.output_dir:
                self.path_output_lbl.configure(text=f"Output: {self.output_dir}")
            if self.backgrounds_dir:
                self.path_backgrounds_lbl.configure(text=f"Backgrounds (no POI): {self.backgrounds_dir}")
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _pick_images_dir(self) -> None:
        d = filedialog.askdirectory(title="Folder containing source images")
        if not d:
            return
        self.images_dir = Path(d).resolve()
        self.path_images_lbl.configure(text=f"Images: {self.images_dir}")
        self._save_paths_json()

    def _pick_labels_dir(self) -> None:
        d = filedialog.askdirectory(title="Folder containing label .txt files")
        if not d:
            return
        self.labels_dir = Path(d).resolve()
        self.path_labels_lbl.configure(text=f"Labels: {self.labels_dir}")
        self._save_paths_json()

    def _pick_output_dir(self) -> None:
        d = filedialog.askdirectory(title="Folder for generated images and labels")
        if not d:
            return
        self.output_dir = Path(d).resolve()
        self.path_output_lbl.configure(text=f"Output: {self.output_dir}")
        self._save_paths_json()

    def _pick_backgrounds_dir(self) -> None:
        d = filedialog.askdirectory(title="Folder with background images (no POI — negatives only)")
        if not d:
            return
        self.backgrounds_dir = Path(d).resolve()
        self.path_backgrounds_lbl.configure(text=f"Backgrounds (no POI): {self.backgrounds_dir}")
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
        """PNG/JPEG/etc. files produced by Extract (under cache assets folder)."""
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
            messagebox.showwarning("Folders", "Choose Images folder, Labels folder, and Output folder.")
            return
        pairs = self._collect_pairs()
        if not pairs:
            messagebox.showinfo(
                "No pairs found",
                "No matching image + .txt pairs.\n\n"
                "Use the same base name for each image and label (e.g. shot.jpg + shot.txt). "
                "Labels are searched recursively in your labels folder.",
            )
            return

        images_dir = self.images_dir
        labels_dir = self.labels_dir
        cache_assets = self._cache_assets_dir()

        self._extracting = True
        self.pipeline_status.configure(
            text=f"Extracting… ({len(pairs)} pair(s); writing crops to output/.poi_studio_cache/assets)",
        )
        self.update_idletasks()

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
                    self._tab_ui_queue.put(
                        ("pipeline", "❌ Extract produced no crops — check labels and image paths."),
                    )
                    return
                self._tab_ui_queue.put(
                    (
                        "pipeline",
                        f"✅ Extract done — {extracted} asset(s) in .poi_studio_cache/assets (run Generate next)",
                    ),
                )
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Extract error: {e}"))
            finally:
                self._tab_ui_queue.put(("extract_done",))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_generate(self) -> None:
        if self._pipeline_busy():
            return
        if not self.output_dir:
            messagebox.showwarning("Folders", "Choose an Output folder first.")
            return
        n_assets = self._count_cached_assets()
        if n_assets == 0:
            messagebox.showwarning(
                "Nothing to paste",
                "Run Extract first so POI crops exist under:\n"
                f"{self._cache_assets_dir()}\n\n"
                "Or change Images/Labels and Extract again.",
            )
            return
        if not self.backgrounds_dir or not self.backgrounds_dir.is_dir():
            messagebox.showwarning(
                "Backgrounds folder",
                "Choose “Backgrounds (no POI)” — a folder of images that do not contain a POI "
                "(screenshots / negatives used only as paste targets).",
            )
            return
        n_bg = self._count_background_images(self.backgrounds_dir)
        if n_bg == 0:
            messagebox.showwarning(
                "No background images",
                f"No supported images (.jpg, .png, …) were found under:\n{self.backgrounds_dir}",
            )
            return
        try:
            n = int(self.synth_count_var.get().strip())
            if n < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid count", "Enter a positive integer.")
            return

        backgrounds_dir = self.backgrounds_dir
        output_dir = self.output_dir
        cache_assets = self._cache_assets_dir()
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"

        self._generating = True
        self.pipeline_status.configure(
            text=f"Generating… ({n_assets} cached asset(s); creating {n} synthetic image(s))",
        )
        self.update_idletasks()

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
                self._tab_ui_queue.put(
                    (
                        "pipeline",
                        f"✅ Done — {n} synthetic image(s) → {out_img} (+ labels in {out_lbl.name}/)",
                    ),
                )
            except Exception as e:
                self._tab_ui_queue.put(("pipeline", f"❌ Generate error: {e}"))
            finally:
                self._tab_ui_queue.put(("generate_done",))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_close(self) -> None:
        if self._tab_ui_poll_id is not None:
            try:
                self.main_app.root.after_cancel(self._tab_ui_poll_id)
            except tk.TclError:
                pass
            self._tab_ui_poll_id = None
