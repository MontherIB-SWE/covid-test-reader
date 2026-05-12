# POI Studio — Comprehensive Project Status Report

**Report Date:** 2026-05-11
**Project Root:** `C:\Users\mndr6\Downloads\archive`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Purpose and Domain](#2-project-purpose-and-domain)
3. [Top-Level Directory Structure](#3-top-level-directory-structure)
4. [Source Code Files — Detailed Analysis](#4-source-code-files--detailed-analysis)
5. [The `studio/` Package — Module-by-Module Breakdown](#5-the-studio-package--module-by-module-breakdown)
6. [Architecture and Component Interaction](#6-architecture-and-component-interaction)
7. [Data Flow and Storage](#7-data-flow-and-storage)
8. [Key Workflows](#8-key-workflows)
9. [Configuration and Persistence](#9-configuration-and-persistence)
10. [Training Infrastructure](#10-training-infrastructure)
11. [Data Assets and Dataset State](#11-data-assets-and-dataset-state)
12. [UI Design and Theme System](#12-ui-design-and-theme-system)
13. [Platform-Specific Features](#13-platform-specific-features)
14. [Dependencies](#14-dependencies)
15. [Current State and Observations](#15-current-state-and-observations)
16. [Known Technical Debt and Notes](#16-known-technical-debt-and-notes)
17. [File Inventory Summary](#17-file-inventory-summary)

---

## 1. Executive Summary

**POI Studio** is a Windows desktop application built with Python/Tkinter that provides a complete pipeline for detecting, labeling, augmenting, and training AI models to find **POI (Point of Interest)** objects in images. The application uses **YOLO OBB (Oriented Bounding Box)** models from the Ultralytics framework to detect quadrilateral regions in images and video streams. It is organized as a three-tab GUI: **Run** (live inference / camera / screen capture), **Data** (labeling + synthetic dataset generation), and **Train** (YOLO model training). A monolithic single-file version (`poi_studio_monolith.py`) also exists as a historical artifact.

---

## 2. Project Purpose and Domain

The project addresses the problem of detecting a specific visual object (referred to as "POI") in images and live video feeds. The workflow is:

1. **Inference**: Run a trained YOLO-OBB model on images, camera streams, or screen captures to detect POI instances as oriented quadrilaterals.
2. **Labeling**: Manually draw or correct 4-corner quadrilateral annotations on images. An AI-assist feature can pre-populate corners using the current model.
3. **Data Augmentation**: Extract POI crops as transparent PNG assets, then paste them onto negative (no-POI) background images with random rotation, scale, and color jitter to synthesize training data.
4. **Training**: Bundle labeled + synthetic data into YOLO-OBB format, split into train/val, and train or fine-tune a YOLOv8s-OBB model.

The entire loop is designed to be iterative — label, augment, train, deploy, and repeat.

---

## 3. Top-Level Directory Structure

```
archive/
├── AGENTS.md                        # AI agent file creation rules
├── CLAUDE.md                        # (Identical to AGENTS.md)
├── opencode.md                      # (Identical to AGENTS.md)
├── poi_studio.py                    # Unified launcher (Run+Data+Train)
├── poi_studio_monolith.py           # Monolithic single-file build (~2300 lines)
├── viewer_app.py                    # Standalone Run-only launcher
├── data_app.py                      # Standalone Data-only launcher
├── train_app.py                     # Standalone Train-only launcher
├── yolo26n.pt                       # YOLO Nano pretrained weights
├── .poi_studio_dataset.json         # Data tab path persistence
├── studio/                          # Main Python package
│   ├── __init__.py                  # Package docstring
│   ├── config.py                    # Constants, theme, Win32 capture exclusion
│   ├── widgets.py                   # Custom Tk widgets (FlatButton, FlatLabel, FlatOptionMenu)
│   ├── viewer.py                    # Run tab — PoiDesktopViewer class
│   ├── relabel.py                   # Label sub-tab — RelabelCornersTool class
│   ├── data_tab.py                  # Data tab — DataManagementTab class
│   ├── train_tab.py                 # Train tab — TrainingTab class
│   ├── pipeline.py                  # Data pipeline (extract, generate, reorganize)
│   ├── shell.py                     # Unified shell — PoiStudio class
│   └── __pycache__/                 # Compiled bytecode (Python 3.13 and 3.14)
├── data_master/                     # Curated training data
│   ├── labeled_with_poi/            # Hand-labeled POI images
│   │   ├── images/                  # (some .jpg files)
│   │   └── labels/                  # 6 .txt label files
│   ├── generated_/                  # Synthetic output (10,000 images)
│   │   ├── images/                  # synth_00000.jpg .. synth_09999.jpg
│   │   └── labels/                  # Matching .txt YOLO OBB labels
│   └── without_poi_negatives/       # Negative backgrounds (no POI)
│       ├── images/
│       └── labels/
├── data/                            # Auto-label output / raw data
│   └── autolabel_output/
│       ├── images/                  # Auto-captured frames (2 .jpg)
│       └── labels/                  # Auto-generated labels (1 .txt)
├── outputs/                         # Training bundle and configuration
│   ├── poi_train_data.yaml          # YOLO dataset YAML for training
│   ├── .poi_train_sources.json      # Train tab source configuration
│   └── .poi_train_bundle/           # Bundled training dataset
│       ├── images/
│       │   ├── train/               # Training images (~1,800+)
│       │   └── val/                 # Validation images (~300+)
│       └── labels/
│           ├── train/               # Training labels
│           └── val/                 # Validation labels
└── runs/                            # YOLO training run outputs
    ├── obb/
    │   ├── runs/
    │   │   ├── obb/
    │   │   │   ├── runs/
    │   │   │   │   └── poi_train/
    │   │   │   │       └── exp/      # Latest training experiment
    │   │   │   │           ├── weights/
    │   │   │   │           │   ├── best.pt
    │   │   │   │           │   └── last.pt
    │   │   │   │           ├── args.yaml
    │   │   │   │           ├── results.csv
    │   │   │   │           ├── results.png
    │   │   │   │           ├── confusion_matrix.png
    │   │   │   │           ├── confusion_matrix_normalized.png
    │   │   │   │           ├── BoxR_curve.png
    │   │   │   │           ├── BoxP_curve.png
    │   │   │   │           ├── BoxF1_curve.png
    │   │   │   │           ├── BoxPR_curve.png
    │   │   │   │           ├── train_batch0.jpg .. train_batch2.jpg
    │   │   │   │           ├── val_batch0_labels.jpg .. val_batch2_pred.jpg
    │   │   │   │           └── labels.jpg
    │   │   ├── obb_v1/
    │   │   │   └── poi_obb_v1/
    │   │   │       ├── weights/ (best.pt, last.pt)
    │   │   │       ├── args.yaml
    │   │   │       └── results.csv
    │   │   └── obb_v7/
    │   │       └── poi_obb_v7/
    │   │           ├── args.yaml
    │   │           └── (weights and results)
    ├── segment/                      # Legacy segmentation experiments
    │   └── runs/
    │       ├── segment_v10/ (best.pt, last.pt, args.yaml, results.csv)
    │       └── segment/
    │           └── outputs/models/poi_segmenter_v2/ (best.pt, last.pt, args.yaml, results.csv)
    └── outputs/ evaluation reports
        ├── evaluation_report_v7.csv
        └── evaluation_report.csv
```

---

## 4. Source Code Files — Detailed Analysis

### 4.1 `poi_studio.py` — Unified Launcher (34 lines)

The primary entry point for the unified application. Creates a `tk.Tk` root window, instantiates `PoiStudio` from `studio.shell`, and starts the mainloop. Includes a crash handler that writes tracebacks to `logs/app.log`.

**Key behavior:**
- Imports `PoiStudio` from `studio.shell`
- Catches all exceptions at the top level and logs them
- Creates a `logs/` directory on crash

### 4.2 `poi_studio_monolith.py` — Monolithic Build (~2,300 lines)

A self-contained single-file version of the entire application containing all classes inline: `FlatButton`, `FlatLabel`, `FlatOptionMenu`, `PoiDesktopViewer`, `DataManagementTab`, `RelabelCornersTool`. This is a historical artifact — the code has since been refactored into the `studio/` package. It contains inline constants, theme definitions, and the complete set of `DataManagementTab` functionality including synthetic generation, pair preview, and a "clear generated outputs" safety review dialog.

**Key difference from modular version:** The monolith's `DataManagementTab` is more feature-rich than the modular `data_tab.py`, containing pair preview with label overlays, thumbnail strips, filmstrip of recent synthetics, generation signal detection (heuristic detection of pipeline-generated files by filename/path patterns), and a comprehensive deletion review modal with file size summaries, extension histograms, and largest-file listings.

### 4.3 `viewer_app.py` — Standalone Viewer Launcher (20 lines)

Minimal host that creates a root window and instantiates only `PoiDesktopViewer` from `studio.viewer`. Useful for inference-only workflows without the Data or Train tabs.

### 4.4 `data_app.py` — Standalone Data Launcher (36 lines)

Hosts only `DataManagementTab` in a standalone window. Provides labeling and dataset preparation without the Run or Train surfaces. Sets `self.trainer = None` since there is no training tab.

### 4.5 `train_app.py` — Standalone Train Launcher (48 lines)

Hosts only `TrainingTab` in a standalone window. After training completes, it shows a messagebox instructing the user to open `viewer_app.py` to load the trained weights. Does not auto-load the model.

---

## 5. The `studio/` Package — Module-by-Module Breakdown

### 5.1 `studio/__init__.py` (1 line)

Docstring only: `"POI Studio package: viewer, data pipeline, training."`

### 5.2 `studio/config.py` — Shared Constants and Theme (114 lines)

Central configuration file defining all shared constants:

| Category | Constants |
|----------|-----------|
| **File extensions** | `SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}` |
| **Model defaults** | `DEFAULT_MODEL_PATH = Path("runs/obb_v1/poi_obb_v1/weights/best.pt")` |
| **Directory paths** | `DATA_MASTER`, `OUTPUTS_DIR`, `RUNS_DIR` |
| **Training defaults** | `DEFAULT_LR=0.002`, `DEFAULT_BATCH=4`, `DEFAULT_EPOCHS=150`, `DEFAULT_IMGSZ=960`, `DEFAULT_PATIENCE=30` |
| **Live capture** | `LIVE_TARGET_FPS=30`, `CAM_DISCONNECT_THRESHOLD=10`, `AUTOLABEL_MIN_INTERVAL=1.0` |
| **POI colors** | `POI_COLORS` (8 BGR tuples), `POI_COLORS_HEX`, `DRAG_RADIUS=8` |
| **Theme colors** | `BG=#0f0f0f`, `BG_PANEL=#1a1a1a`, `BG_CARD=#222222`, `BG_INPUT=#2a2a2a`, `BG_HOVER=#333333`, `BG_ACTIVE=#3a3a3a`, `FG=#e0e0e0`, `FG_DIM=#888888`, `ACCENT=#00a86b`, `ORANGE=#e0952a`, `RED=#e04040`, `YELLOW=#e0c040`, `CYAN=#40c0e0` |
| **Fonts** | `FONT="Segoe UI"`, `MONO="Consolas"` |

Also contains Win32 `SetWindowDisplayAffinity` bindings for screen capture exclusion (Windows 10 2004+), and a `_select_device()` function that returns `"cuda"` if available, else `"cpu"`.

### 5.3 `studio/widgets.py` — Custom Themed Tk Widgets (163 lines)

Three custom widget classes:

**`FlatButton(tk.Canvas)`** — A fully custom button drawn as a rounded rectangle on a Canvas. Supports hover color change, configurable radius, bold text, and dynamic text/color updates via `configure_text()` and `configure_colors()`. Uses smooth polygon points for rounded corners.

**`FlatLabel(tk.Label)`** — Thin wrapper around `tk.Label` with default dark theme styling (`bg=BG_PANEL`, `fg=FG`, `font=(FONT, 9)`).

**`FlatOptionMenu(tk.Frame)`** — Custom dropdown menu implementation using a `Toplevel` overlay window. Does not use native Tk OptionMenu. Supports dynamic value updates via `update_values()`. Shows a floating list of options with hover highlight.

### 5.4 `studio/viewer.py` — Run Tab / Inference Viewer (993 lines)

The `PoiDesktopViewer` class is the core inference surface. Architecture:

**State management:**
- `model: YOLO | None` — Currently loaded YOLO model
- `model_path: Path` — Path to the .pt weights file
- `device: str` — "cuda" or "cpu"
- `live_running: bool` — Whether the live feed is active
- `autolabel_active: bool` — Whether auto-labeling is saving frames
- `active_live_source: str | None` — "Camera" or "Screen"

**Threading model:**
- A dedicated **camera reader thread** (`_camera_reader_loop`) continuously reads frames from `cv2.VideoCapture` into a shared `_latest_frame` buffer protected by `_latest_frame_lock`
- A separate **inference worker thread** per frame runs `model.predict()` and pushes results to `_live_ui_queue`
- The **main thread** polls `_live_ui_queue` via `_live_tick()` at ~30 FPS using `root.after()`, draining results and updating the display
- Camera scanning runs in a background thread with results delivered via `_camera_results_queue`

**Key methods:**
- `_build_ui()` — Constructs the header toolbar, viewport (image overlay + crop grid), and status bar
- `_run_prediction(image_path)` — Synchronous prediction on a single image file
- `_start_live()` / `_stop_live()` — Start/stop the live inference loop
- `_live_tick()` — Main loop: reads frame, dispatches inference worker, drains UI queue
- `_build_outputs(bgr, result)` — Processes YOLO OBB results: draws colored overlays, generates perspective-warped crops, builds status text
- `_order_quad_points(pts)` — Orders 4 corner points as [TL, TR, BR, BL] using sum/diff heuristic
- `_warp_quad_crop(bgr, quad)` — Applies perspective transform to extract a deskewed crop of each POI
- `_toggle_autolabel()` — Starts/stops automatic label saving
- `_save_prediction_as_label(bgr, result)` — Writes image + YOLO OBB label file (empty for negatives)
- `_capture_for_relabel()` — Captures current frame and triggers the relabel callback

**Screen capture features:**
- When source is "Screen", uses `PIL.ImageGrab.grab()` to capture the desktop
- Can exclude its own window from capture using Win32 `SetWindowDisplayAffinity`

**Auto-label pipeline:**
- When active, saves every frame with detected POIs to `data/autolabel_output/images/` with matching labels in `data/autolabel_output/labels/`
- Throttled to 1 frame per second (`AUTOLABEL_MIN_INTERVAL`)
- Negative frames (no POI) get empty label files

### 5.5 `studio/relabel.py` — Labeling Tool (1,053 lines)

The `RelabelCornersTool` class provides an interactive polygon editor for POI corner annotation. Architecture:

**Data model:**
- `EditablePoi` — Holds a list of `(x, y)` pixel-coordinate points and a `color_idx`
- `pois: list[EditablePoi]` — All POIs for the current image
- `new_points_canvas: list[tuple[int,int]]` — In-progress new POI corners

**Canvas interaction:**
- Left-click: Select POI (if near a point, start drag; if inside a polygon, select it; otherwise add a new corner point)
- Right-click: Delete a single point from a POI (minimum 3 points required)
- Mouse drag: Move corner points
- Scroll wheel: Zoom in/out at cursor position
- Middle-click drag: Pan the canvas
- Double-click: Reset zoom

**Label format:** YOLO OBB format — `class x1 y1 x2 y2 x3 y3 x4 y4` where coordinates are normalized (0-1).

**Key features:**
- Zoom/pan with configurable limits (0.5x–16x)
- Auto-save on every change (writes label file immediately)
- POI list panel on the left showing all POIs with color indicators
- Preview canvas on the right showing OpenCV-rendered result
- AI Assist: Calls back to the viewer's model to auto-populate POI corners
- Save-to-train callback: Moves labeled images to `data_master/labeled_with_poi/`
- Delete label file function with confirmation dialog

**Callbacks (set by the shell):**
- `_save_to_train_callback` — Moves image+label to the master training directory
- `_ai_assist_callback` — Runs inference on the current image and returns detected corners

### 5.6 `studio/data_tab.py` — Data Management Tab (570 lines)

The `DataManagementTab` class provides the Data surface with two sub-tabs:

**Label sub-tab:** Hosts an embedded `RelabelCornersTool` instance.

**Dataset sub-tab:** A simplified synthetic generation pipeline:
1. Pick Images folder (source images with POI)
2. Pick Labels folder (matching .txt YOLO labels)
3. Pick Backgrounds folder (negative images without POI)
4. Pick Output folder (destination for synthetics)
5. **Extract** — Crops POI assets as transparent PNGs into `output/.poi_studio_cache/assets/`
6. **Generate** — Pastes assets onto backgrounds with random augmentation -> writes `output/images/synth_*.jpg` + `output/labels/synth_*.txt`

**Path persistence:** Saves folder paths to `.poi_studio_dataset.json` in the working directory. Restores on next launch.

**Thread safety:** Extract and Generate run in background threads. UI updates are marshalled to the main thread via `_tab_ui_queue` polled every 120ms.

### 5.7 `studio/pipeline.py` — Data Processing Pipeline (920 lines)

Contains three merged modules that form the data processing pipeline:

#### 5.7.1 Asset Extraction (`extract_assets`)

Reads labeled images + YOLO OBB labels, crops each POI polygon with padding, creates an eroded alpha mask, and saves as BGRA PNG. Also saves a sidecar `.txt` with 4 corner coordinates (pixel-space relative to crop).

Key function: `extract_assets(images_dirs, labels_dirs, output_dir) -> dict`

Processing steps per POI:
1. Parse YOLO OBB label (8 floats = 4 corners)
2. Create filled polygon mask
3. Erode mask by 3px to clean edges
4. Crop image + mask with 12px padding
5. Convert to BGRA, set alpha from mask
6. Fill transparent pixel BGR with nearest opaque color (defringing)
7. Save PNG + corner sidecar .txt

#### 5.7.2 Synthetic Generation (`generate`)

Pastes transparent POI assets onto negative background images with augmentation:

**Augmentation pipeline per POI:**
1. Random scale (0.5x–1.5x, clamped to 50% of target size)
2. Random color jitter (brightness +/-20%, contrast +/-20%)
3. Random rotation (-45 deg to +45 deg)
4. Defringing: Replace transparent pixel BGR with nearest opaque color (2-pass, 8-neighbour)
5. Random placement with collision avoidance (72 tries, 10px gap between POI bboxes)
6. Alpha compositing with 7x7 Gaussian blur feathering

**Multi-POI:** Each image gets 1-3 POIs with weights [60%, 30%, 10%].

**Output:** `synth_NNNNN.jpg` + `synth_NNNNN.txt` in YOLO OBB format.

**Continuation:** Scans existing output to find the next index, allowing incremental generation.

#### 5.7.3 Dataset Reorganization (`reorganize_dataset_main`)

A one-shot script to reorganize raw data into a clean structure. Defines 10 training sources and 2 test data sources, each with a namespace prefix to avoid filename collisions. Splits into 80% train / 10% val / 10% test. Supports dry-run mode.

### 5.8 `studio/train_tab.py` — Training Tab (996 lines)

The `TrainingTab` class manages the YOLO-OBB training pipeline.

**Two data modes:**

1. **Manual folder pairs** — User adds image/label folder pairs manually. All pairs are combined, shuffled, and split into train/val.

2. **data_master mix** — Automated mixing:
   - Collects all POI-labeled images from `generated_root` and `labeled_poi_root`
   - Deduplicates by resolved path
   - Samples negatives from `without_poi_negatives` to reach target labeled fraction (default 20% POI / 80% negatives)
   - Splits into train/val (85%/15%)

**Bundle preparation:**
- `prepare_train_bundle()` — For manual pairs: copies into `outputs/.poi_train_bundle/images/{train,val}` + matching labels
- `prepare_mixed_data_master_bundle()` — For mix mode: uses all POI images + sampled negatives
- Writes `outputs/poi_train_data.yaml` with absolute paths

**Training execution:**
- Runs in a background thread via `_worker()`
- Creates a `YOLO` model from `yolov8s-obb.pt` (fresh) or user-selected `.pt` (retrain)
- Calls `model.train()` with extensive augmentation parameters:
  - HSV augmentation: h=0.02, s=0.8, v=0.5
  - Geometric: degrees=15, translate=0.2, scale=0.9, shear=5.0, fliplr=0.5, flipud=0.1
  - Mosaic=0.5, auto_augment=randaugment, dropout=0.15
  - Box loss weight=7.5, cls loss weight=0.5, angle loss weight=1.0
  - Warmup=3 epochs, patience=30

**UI updates:** Thread-safe via `_tab_ui_queue` polled every 200ms. Shows indeterminate progress bar during training. After completion, offers to load the new `best.pt` into the Run tab.

**Persistence:** Saves all configuration to `outputs/.poi_train_sources.json`.

### 5.9 `studio/shell.py` — Unified Application Shell (178 lines)

The `PoiStudio` class ties all three tabs together:

**Tab management:**
- Creates three tab frames (Run, Data, Train) with custom tab bar buttons
- Only one tab visible at a time via `_switch_tab()` (pack/pack_forget)
- Switching away from Run tab auto-stops live feed

**Cross-tab integration:**
- `viewer.set_capture_for_relabel_callback(self._open_capture_in_relabel)` — Run tab's "Capture -> Relabel" button switches to Data tab and opens the image in the relabel tool
- `relabel.set_save_to_train_callback(self._save_relabel_to_train)` — Relabel tool's save moves files to `data_master/labeled_with_poi/`
- `relabel.set_ai_assist_callback(self._predict_for_relabel)` — AI Assist runs inference on the Run tab's model and returns corners

**Keyboard shortcuts (scoped to active tab):**
- Run tab: Left/Right (navigate), Space (toggle auto-label)
- Data tab: Ctrl+S (save), Ctrl+Z (undo), Delete, Escape, Return, A (add POI), +/-/Ctrl+0 (zoom)

**Model loading:** `load_trained_model()` method safely loads a trained .pt into the viewer after training completes.

---

## 6. Architecture and Component Interaction

```
+--------------------------------------------------------------+
|                        poi_studio.py                         |
|                      (Entry Point)                           |
|                          |                                   |
|                    +-----v------+                            |
|                    |  shell.py  |                            |
|                    | PoiStudio  |                            |
|                    +-----+------+                            |
|            +-------------+-------------+                     |
|     +------v------+ +----v-----+ +-----v------+            |
|     |  viewer.py  | |data_tab.py| | train_tab.py|           |
|     | PoiDesktop  | | DataMgmt  | | TrainingTab |           |
|     |   Viewer    | |   Tab     | |             |           |
|     +------+------+ +----+-----+ +-----+------+            |
|            |              |              |                    |
|            |        +-----v------+  +----v------+           |
|            |        |relabel.py  |  |pipeline.py |           |
|            |        |RelabelCorner|  |(extract,   |           |
|            |        |   Tool     |  | generate,  |           |
|            |        +------------+  | reorganize)|           |
|            |                        +------------+           |
|     +------v------------------------------------+           |
|     |           config.py                        |           |
|     |  (constants, theme, Win32 capture)         |           |
|     +--------------------------------------------+           |
|     +--------------------------------------------+           |
|     |           widgets.py                        |           |
|     |  (FlatButton, FlatLabel, FlatOptionMenu)    |           |
|     +--------------------------------------------+           |
+--------------------------------------------------------------+
```

**Communication patterns:**

1. **Shell -> Tabs:** Direct method calls (e.g., `self.viewer._stop_live()`, `self.data_tab_shell.focus_label_tab()`)
2. **Tabs -> Shell:** Via callback functions set by the shell (e.g., `_save_to_train_callback`, `_ai_assist_callback`, `_capture_for_relabel_callback`)
3. **Background -> UI:** Thread-safe queue pattern (`queue.Queue` + `root.after()` polling)
4. **Relabel -> Viewer model:** AI Assist calls `shell._predict_for_relabel()` which calls `viewer.model.predict()`
5. **Train -> Viewer:** After training, `shell.load_trained_model()` updates `viewer.model_path` and calls `viewer._load_model()`

---

## 7. Data Flow and Storage

### 7.1 Input Data Sources

| Source | Path Pattern | Content |
|--------|-------------|---------|
| Labeled POI images | `data_master/labeled_with_poi/images/*.jpg` | Images containing POI |
| Labeled POI labels | `data_master/labeled_with_poi/labels/*.txt` | YOLO OBB labels (4 corners) |
| Negative images | `data_master/without_poi_negatives/images/` | Images without POI |
| Generated images | `data_master/generated_/images/synth_*.jpg` | Synthetic POI images |
| Generated labels | `data_master/generated_/labels/synth_*.txt` | Synthetic YOLO OBB labels |
| Auto-label output | `data/autolabel_output/{images,labels}/` | Auto-captured frames |
| Pretrained model | `yolo26n.pt` | YOLO Nano weights |
| Training weights | `runs/.../best.pt` | Trained model weights |

### 7.2 Output Data Destinations

| Destination | Path Pattern | Written By |
|-------------|-------------|-----------|
| Training bundle | `outputs/.poi_train_bundle/images/{train,val}/` | `train_tab.py` |
| Training labels | `outputs/.poi_train_bundle/labels/{train,val}/` | `train_tab.py` |
| Dataset YAML | `outputs/poi_train_data.yaml` | `train_tab.py` |
| Training config | `outputs/.poi_train_sources.json` | `train_tab.py` |
| Data paths | `.poi_studio_dataset.json` | `data_tab.py` |
| Synthetic assets | `output/.poi_studio_cache/assets/*.png` | `pipeline.py::extract_assets()` |
| Synthetic images | `output/images/synth_*.jpg` | `pipeline.py::generate()` |
| Synthetic labels | `output/labels/synth_*.txt` | `pipeline.py::generate()` |
| Training runs | `runs/obb/runs/poi_train/exp/` | YOLO `model.train()` |
| Auto-label frames | `data/autolabel_output/images/` | `viewer.py` |
| Crash log | `logs/app.log` | `poi_studio.py` |

### 7.3 Label Format

YOLO OBB format — one line per object:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
Where:
- `class_id` = `0` (always "poi")
- `x1,y1 ... x4,y4` = Normalized (0-1) coordinates of 4 corners
- Negative samples have empty `.txt` files (0 bytes)

---

## 8. Key Workflows

### 8.1 Live Inference Workflow

```
Start -> Open camera/screen -> _live_tick() loop:
  1. Read frame (from camera thread or screen grab)
  2. If no inference pending, dispatch worker thread:
     - model.predict(frame, conf=0.50, imgsz=960)
     - Build overlay + crops
     - Push to _live_ui_queue
  3. Drain queue on main thread:
     - Update overlay image
     - Update crop cards
     - If auto-label active, save frame + labels
  4. Schedule next tick (~33ms)
```

### 8.2 Labeling Workflow

```
Open images folder -> Load image -> Load matching label file:
  1. Parse YOLO OBB text -> polygon points (normalized -> pixels)
  2. Display image on canvas with polygon overlays
  3. User interactions:
     - Click polygon: select POI
     - Drag corner: move point (auto-saves)
     - Click empty area: add new corner (4th completes POI)
     - Right-click point: remove point (min 3)
     - AI Assist: predict -> replace all POIs
  4. Auto-save: write updated label file on every change
  5. Ctrl+S: additionally copy to data_master/labeled_with_poi/
```

### 8.3 Synthetic Generation Workflow

```
Step A - Extract:
  labeled_with_poi/images + labels -> extract_assets()
  -> synthetic_assets/*.png (BGRA crops with alpha mask)
  -> synthetic_assets/*.txt (corner coordinates)

Step B - Generate:
  synthetic_assets/ + without_poi_negatives/images -> generate()
  For each output image:
    1. Pick random background -> resize to 960x960
    2. Decide 1-3 POIs per image
    3. For each POI slot (up to 48 tries):
       - Pick random asset
       - Random scale (0.5-1.5x)
       - Random color jitter
       - Random rotation (-45 deg to +45 deg)
       - Defringe transparent edges
       - Try random placement (avoiding overlap)
       - Alpha-composite onto background
    4. Save synth_NNNNN.jpg + .txt
```

### 8.4 Training Workflow

```
1. Configure data sources:
   Option A: Manual folder pairs
   Option B: data_master mix (generated + labeled + negatives at 20/80 ratio)

2. Prepare bundle:
   - Collect all image+label pairs
   - Deduplicate by resolved path
   - Shuffle with seed=42
   - Split 85% train / 15% val
   - Copy into outputs/.poi_train_bundle/
   - Write poi_train_data.yaml

3. Train:
   - Load YOLO model (yolov8s-obb.pt or custom .pt)
   - model.train(data=yaml, epochs=150, batch=4, imgsz=960, ...)
   - Outputs: runs/obb/runs/poi_train/exp/weights/{best,last}.pt

4. Deploy:
   - Offer to load best.pt into viewer
   - Or use "Load Last" button in viewer
```

---

## 9. Configuration and Persistence

### 9.1 `.poi_studio_dataset.json` (Data Tab)

```json
{
  "images_dir": "C:\\...\\data_master\\labeled_with_poi\\images",
  "labels_dir": "C:\\...\\data_master\\labeled_with_poi\\labels",
  "backgrounds_dir": "C:\\...\\data_master\\without_poi_negatives\\images",
  "output_dir": "C:\\...\\data_master\\generated_"
}
```

### 9.2 `outputs/.poi_train_sources.json` (Train Tab)

```json
{
  "pairs": [[images_dir, labels_dir], ...],
  "mix": {
    "use": true,
    "generated": "...",
    "labeled_with_poi": "...",
    "negatives": "...",
    "labeled_pct": "20"
  }
}
```

### 9.3 `outputs/poi_train_data.yaml` (YOLO Dataset)

```yaml
path: C:/Users/mndr6/Downloads/archive/outputs/.poi_train_bundle
train: images/train
val: images/val
names:
  0: poi
```

### 9.4 `runs/.../args.yaml` (Training Arguments)

Full YOLO training configuration with all hyperparameters. Currently shows:
- Model: `yolov8s-obb.pt`
- Device: `0` (CUDA GPU 0)
- Epochs: 150
- Batch: 4
- Image size: 960
- LR: 0.002
- Extensive augmentation parameters

---

## 10. Training Infrastructure

### 10.1 Completed Training Runs

| Run | Path | Model | Status |
|-----|------|-------|--------|
| OBB v1 | `runs/obb/runs/obb_v1/poi_obb_v1/` | Unknown | Completed (has best.pt, results.csv) |
| OBB v7 | `runs/obb/runs/obb_v7/poi_obb_v7/` | Unknown | Completed (has args.yaml) |
| Latest | `runs/obb/runs/obb/runs/obb/runs/poi_train/exp/` | yolov8s-obb | Completed (has best.pt, last.pt, plots, results.csv) |
| Segment v10 | `runs/segment/runs/segment_v10/` | Unknown | Completed |
| Segment v2 | `runs/segment/runs/segment/outputs/models/poi_segmenter_v2/` | Unknown | Completed |

The deeply nested path `runs/obb/runs/obb/runs/obb/runs/poi_train/exp/` suggests the training was run from within a directory that already contained `runs/obb/`, causing YOLO to nest further.

### 10.2 Pretrained Weights

- `yolo26n.pt` — YOLO Nano model at the project root (possibly used as a base or for testing)
- Default training uses `yolov8s-obb.pt` (downloaded automatically by Ultralytics)

### 10.3 Training Output Artifacts

The latest experiment includes:
- `best.pt` / `last.pt` — Model weights
- `results.csv` — Per-epoch metrics
- `results.png` — Training curves plot
- `confusion_matrix.png` / `confusion_matrix_normalized.png`
- `BoxR_curve.png`, `BoxP_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png` — Precision/Recall curves
- `train_batch{0,1,2}.jpg` — Training batch visualizations
- `val_batch{0,1,2}_{labels,pred}.jpg` — Validation predictions vs ground truth
- `labels.jpg` — Label distribution

---

## 11. Data Assets and Dataset State

### 11.1 Labeled POI Data

`data_master/labeled_with_poi/labels/` contains 6 label files:
- `images_1777559152625.txt`
- `images_1777557930792.txt`
- `test_auto_1777512117052.txt`
- `test_neg_IMG_4561.txt`
- `test_bad_IMG_5536.txt`
- `test_bad_IMG_5543.txt`

This is a very small hand-labeled set.

### 11.2 Generated Synthetic Data

`data_master/generated_/` contains **10,000 synthetic images** (synth_00000 through synth_09999) with matching labels. This is the primary training data source.

### 11.3 Training Bundle

`outputs/.poi_train_bundle/` contains the latest prepared training dataset:
- **labels/val/** — At least 2,000+ validation label files (visible up to index ~1961)
- **labels/train/** — Corresponding training labels
- **Caches:** `train.cache` and `val.cache` files (YOLO label cache)

The bundle was prepared using the "mix" mode with 20% labeled / 80% negatives.

### 11.4 Auto-label Output

`data/autolabel_output/` contains 2 auto-captured frames and 1 label file — indicating the auto-label feature has been used but only sparingly.

---

## 12. UI Design and Theme System

The application uses a **dark theme** with the following design language:

**Color palette:**
- Background: Near-black (`#0f0f0f` to `#222222` layered)
- Accent: Emerald green (`#00a86b`)
- Warning/active: Orange (`#e0952a`)
- Danger: Red (`#e04040`)
- Text: Light gray (`#e0e0e0`)
- Dim text: Medium gray (`#888888`)

**Widget system:**
- All buttons are custom `FlatButton` (Canvas-based rounded rectangles)
- Dropdowns are custom `FlatOptionMenu` (Toplevel popup-based)
- Consistent padding, fonts, and border styling

**Layout patterns:**
- Header toolbar (54px height) with horizontally arranged controls
- Status bar at bottom (40px height) with monospace text
- Split-panel viewports with scrollable content areas
- Section cards with colored accent rails

**Fonts:**
- UI: Segoe UI (9-15pt)
- Monospace: Consolas (8-11pt)

---

## 13. Platform-Specific Features

### 13.1 Windows Screen Capture Exclusion

Uses Win32 API `SetWindowDisplayAffinity` with `WDA_EXCLUDEFROMCAPTURE` to hide the application window from screen captures (useful when the live source is "Screen" to avoid recursive mirror effects).

Implementation in `config.py`:
- Loads `user32.dll` via `ctypes.WinDLL`
- Walks up from widget HWND to top-level window using `GetAncestor(hwnd, GA_ROOT)`
- Gracefully falls back if not on Windows 10 2004+

### 13.2 Camera Support

- Uses DirectShow (`cv2.CAP_DSHOW`) on Windows for camera capture
- Redirects C-level stderr (fd 2) during camera probing to suppress DirectShow assertion noise
- Auto-scans camera indices 0-2 on startup

### 13.3 Unicode Path Handling

`pipeline.py` includes `_cv_imread_unicode()` and `_cv_imwrite_unicode()` that fall back to `numpy.fromfile()` + `cv2.imdecode()` for Windows Unicode path support (cv2.imread doesn't handle non-ASCII paths on Windows).

---

## 14. Dependencies

Based on imports across all source files:

| Package | Usage |
|---------|-------|
| `tkinter` | GUI framework (standard library) |
| `cv2` (OpenCV) | Image processing, camera capture, polygon drawing, perspective warp |
| `numpy` | Array operations, coordinate transformations |
| `PIL` (Pillow) | Image conversion, thumbnails, screen capture (`ImageGrab`) |
| `ultralytics` (YOLO) | Model loading, inference, training (`YOLO` class) |
| `torch` (PyTorch) | CUDA detection, YOLO backend (transitive via ultralytics) |
| `ctypes` | Win32 API calls (standard library) |
| `threading` | Background workers, camera reader thread |
| `queue` | Thread-safe UI communication |
| `json` | Configuration persistence |
| `shutil` | File copy/move operations |
| `random` | Data shuffling, augmentation randomness |
| `argparse` | CLI argument parsing (pipeline.py) |
| `math`, `time`, `os`, `re`, `collections` | Various utilities |

---

## 15. Current State and Observations

### 15.1 Active and Working

- **Run tab** is fully functional with image, camera, and screen capture inference
- **Data tab** has a working labeling tool with AI assist, auto-save, and zoom/pan
- **Train tab** successfully completed at least 3 training runs (OBB v1, v7, and latest)
- **Synthetic generation** has produced 10,000 images, indicating the pipeline is mature
- **Training bundle** is prepared with ~2,000+ samples in validation alone

### 15.2 Dual Data Tab Implementations

There are two versions of `DataManagementTab`:

1. **`studio/data_tab.py`** (570 lines) — Simpler version with basic Extract/Generate pipeline and no pair preview
2. **`poi_studio_monolith.py`** (~1,500+ lines for DataManagementTab) — Feature-rich version with:
   - Pair preview with label overlays
   - Thumbnail strips for loaded pairs
   - Filmstrip of recent synthetic images
   - Generation signal detection (heuristic flags for pipeline-generated files)
   - Comprehensive deletion review modal
   - Working directory configuration
   - Three-step pipeline UI (Folders -> Build -> Split)

The modular `data_tab.py` is a simplified rewrite that loses some features from the monolith. The monolith's DataManagementTab has a more complete dataset management experience.

### 15.3 Monolith vs Modular

The `poi_studio_monolith.py` file contains the entire application in a single file (~2,300 lines) with inline widget classes and all the same viewer/relabel functionality. This appears to be the original version before the refactoring into `studio/` package modules. The monolith is kept for reference but is no longer the primary way to run the app.

### 15.4 AGENTS.md Inconsistency

The `AGENTS.md` file references `desktop_poi_viewer.py` and `relabel_corners_tool.py` as the main files, but these files do not exist in the project. The actual files are `poi_studio.py` (launcher), `studio/viewer.py`, and `studio/relabel.py`. The AGENTS.md rules about file creation still apply conceptually.

---

## 16. Known Technical Debt and Notes

1. **Deeply nested run path:** `runs/obb/runs/obb/runs/obb/runs/poi_train/exp/` — This is 4 levels of nesting because YOLO was run from within a directory that already had `runs/obb/`. The `RUNS_DIR = Path("runs/obb/runs")` constant combined with YOLO's own `runs/` prefix creates the nesting.

2. **Default model path mismatch:** `config.py` sets `DEFAULT_MODEL_PATH = Path("runs/obb_v1/poi_obb_v1/weights/best.pt")`, but this path doesn't exist (the actual path is `runs/obb/runs/obb_v1/poi_obb_v1/weights/best.pt`). This means the viewer starts without a loaded model.

3. **Duplicate AGENTS.md/CLAUDE.md/opencode.md:** All three files have identical content. This is redundant.

4. **`__pycache__` for Python 3.13 and 3.14:** Both bytecode versions exist, suggesting the project has been run under two Python versions.

5. **Missing `data_master/labeled_with_poi/images/` files:** Only 6 label files exist in the labels folder, but the corresponding image files were not enumerated in the glob results (the glob returned label .txt files but the images directory contents were truncated). The small number suggests this is a seed dataset that has been mostly superseded by synthetic data.

6. **Segmentation models are legacy:** The `runs/segment/` directory contains older segmentation experiments, but the current codebase is entirely focused on OBB detection. These are historical artifacts.

7. **Thread safety in pipeline.py:** The `generate()` function modifies module-level globals (`BG_DIR`, `ASSETS_DIR`, `OUT_IMG_DIR`, `OUT_LBL_DIR`) before running. While functional in single-window use, this is not thread-safe if multiple generation runs were triggered simultaneously.

8. **No requirements.txt or setup.py:** The project has no dependency management file. Dependencies must be installed manually.

---

## 17. File Inventory Summary

### Source Code Files (14 files)

| File | Lines | Purpose |
|------|-------|---------|
| `poi_studio.py` | 34 | Unified launcher |
| `poi_studio_monolith.py` | ~2,300 | Monolithic legacy build |
| `viewer_app.py` | 20 | Standalone viewer launcher |
| `data_app.py` | 36 | Standalone data launcher |
| `train_app.py` | 48 | Standalone train launcher |
| `studio/__init__.py` | 1 | Package init |
| `studio/config.py` | 114 | Constants, theme, Win32 |
| `studio/widgets.py` | 163 | Custom UI widgets |
| `studio/viewer.py` | 993 | Run tab / inference |
| `studio/relabel.py` | 1,053 | Labeling tool |
| `studio/data_tab.py` | 570 | Data management tab |
| `studio/train_tab.py` | 996 | Training tab |
| `studio/pipeline.py` | 920 | Data pipeline |
| `studio/shell.py` | 178 | Unified shell |
| **Total** | **~7,456** | |

### Configuration Files (9 YAML + 2 JSON)

| File | Purpose |
|------|---------|
| `.poi_studio_dataset.json` | Data tab folder paths |
| `outputs/.poi_train_sources.json` | Train tab configuration |
| `outputs/poi_train_data.yaml` | YOLO dataset descriptor |
| `runs/.../args.yaml` (x6) | Training run configurations |

### Documentation Files (3, identical)

- `AGENTS.md` / `CLAUDE.md` / `opencode.md`

### Data Assets

| Category | Approximate Count |
|----------|------------------|
| Labeled POI labels | 6 files |
| Generated synthetic images | 10,000 |
| Generated synthetic labels | 10,000 |
| Training bundle (val labels) | ~2,000+ |
| Training bundle (train labels) | ~8,000+ (estimated) |
| Auto-label output | 2 images, 1 label |
| Trained model weights | 8 .pt files (4 runs x best+last) |
| Evaluation reports | 2 CSV files |
| Pretrained weights | 1 file (`yolo26n.pt`) |

### Bytecode Cache

- `studio/__pycache__/` — 12 .pyc files (6 for Python 3.13, 6 for Python 3.14)

---

*End of Report*
