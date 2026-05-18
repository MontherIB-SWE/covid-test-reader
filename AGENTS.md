# AI Agent Rules

## File Creation Policy
ALWAYS ask the user before creating new files.

## Rules
1. **Ask before creating** - Before creating any new .py file, ask: "Should I create a new file for this?"
2. **Prefer extending** - Suggest adding to existing `studio/` modules first (especially `studio/qt/` for UI)
3. **No one-time scripts** - If a temporary script is needed, ask where to put it and clean up after
4. **Centralize paths** - All paths are defined in `studio/config.py`. Never hardcode paths elsewhere.
5. **Explain first** - Before creating anything, explain what you want to create and why
6. **User decides** - The user's answer determines whether to proceed with file creation
7. **Monolith is frozen** - `poi_studio_monolith.py` is the legacy single-file version. Do not modify it.

## Project Structure

### Entry points
- `poi_studio.py` - Main launcher (PySide6 `studio.qt.shell.run_app()`)
- `poi_studio_monolith.py` - FROZEN legacy Tk monolith (do not modify)

### studio/ package
- `studio/config.py` - All paths, constants, theme, Win32 helpers
- `studio/pipeline.py` - Synthetic data pipeline (extract assets, generate composites)
- `studio/train_ops.py` - Training bundle prep (no UI; used by Train tab)
- `studio/qt/` - **PySide6 UI**
  - `studio/qt/shell.py` - `PoiStudioWindow`, `run_app()`
  - `studio/qt/viewer_tab.py` - Run tab (inference, live camera/screen)
  - `studio/qt/relabel_widget.py` - Label sub-tab (corner editor)
  - `studio/qt/data_tab.py` - Data tab (Label + Dataset)
  - `studio/qt/train_tab.py` - Train tab
  - `studio/qt/theme.py` - Application QSS
  - `studio/qt/images.py`, `studio/qt/widgets.py`, `studio/qt/win32.py` - helpers

### Data directories
- `data/labeled/` - Hand-labeled POI images (images/ + labels/)
- `data/negatives/` - Background images without POI (images/ + labels/)
- `data/generated/` - Synthetic composites (images/ + labels/ + .poi_studio_cache/)
- `data/unlabeled/` - Unlabeled POI images (not yet labeled)
- `data/autolabel/` - Auto-captured frames
- `data/assets/` - Extracted POI crops (PNG with alpha)

### Build artifacts
- `outputs/` - Training bundles, JSON state, dataset YAML
- `runs/` - YOLO training runs (weights, metrics)

## Dependencies
See `requirements.txt` (PySide6, ultralytics, opencv-python-headless, numpy, Pillow).

## Path Architecture
All paths are defined once in `studio/config.py` as module-level constants:
- `DATA_DIR`, `DATA_LABELED_DIR`, `DATA_NEGATIVES_DIR`, etc.
- `OUTPUTS_DIR`, `TRAIN_SOURCES_JSON`, `TRAIN_BUNDLE_DIR`, etc.
- `RUNS_DIR`, `DEFAULT_MODEL_PATH`
- `resolve_model_path()` finds latest best.pt in runs/

Every module imports from `studio.config` — no hardcoded paths elsewhere.

## Manual regression (after UI changes)
1. Run: open image, run prediction, Start/Stop live (camera and screen on Windows), auto-label toggle, Capture → Relabel.
2. Data: Label tab open folder, edit POIs, save / Ctrl+S export, Dataset Extract + Generate.
3. Train: mix or manual pairs, start training (smoke), Open runs folder; on success confirm load model switches to Run tab.
4. Shortcuts: Left/Right by tab; Data tab zoom (scroll), pan (middle drag), double-click reset.

## Before Creating Any File
1. Does this already exist?
2. Can it be added to an existing file?
3. Have I asked the user for permission?

If "no" to any: STOP and ASK.
