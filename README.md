# COVID Test Reader

Desktop app that **finds a COVID-19 rapid antigen test in an image or live camera feed and reads its result** — `positive`, `negative`, or `invalid`.

Built as my Machine Learning Lab project (final grade: **100**). Two-stage vision pipeline plus the full data tooling needed to train it: labeling, augmentation, training, and live inference in one PySide6 app.

---

## How it works

**Stage 1 — Detection.** A YOLOv8s **OBB** (oriented bounding box) model locates the test cassette as a rotated quadrilateral, so the test can be held at any angle rather than square to the camera.

**Stage 2 — Classification.** The detected quad is perspective-warped into a rectified, letterboxed crop of the result window, then a fine-tuned **ResNet50** (ImageNet-normalized) classifies it as `positive` / `negative` / `invalid`.

Splitting detection from classification means the classifier only ever sees a clean, upright result window — lighting, angle, and background are handled upstream.

```
image / camera frame
        ↓
  YOLOv8s-OBB  ──→  oriented quad around the cassette
        ↓
  perspective warp + letterbox
        ↓
     ResNet50  ──→  positive | negative | invalid
```

## The data problem

Rapid-test photos are scarce, so most of the work was building a dataset engine rather than a model:

- **Labeling UI** — draw or correct the 4 corners by hand, with AI-assisted pre-fill from the current model.
- **Asset extraction** — labeled cassettes are cut out as transparent PNGs with an eroded alpha mask.
- **Synthetic generation** — those cutouts are composited onto negative (no-test) backgrounds with random rotation, scale, and color jitter.
- **Classifier augmentation** — 12 variants per source crop; 15% validation split, fixed seed.
- **Active learning** — corrections made during live inference are logged to JSONL and folded back into the next training run.

Current hand-labeled result set: **1,743 crops** — 848 negative, 767 positive, 128 invalid.

## The app

PySide6 desktop app, three tabs:

| Tab | What it does |
|---|---|
| **Run** | Live inference at ~30 FPS on webcam, screen capture, or still images |
| **Data** | Label corners, extract assets, generate synthetic data, manage the manifest |
| **Train** | Bundle to YOLO-OBB format, split train/val, train or fine-tune both models |

## Stack

`Python` · `PyTorch` · `Ultralytics YOLOv8-OBB` · `ResNet50` · `OpenCV` · `PySide6 (Qt)` · `NumPy`

## Running it

```bash
pip install -r requirements.txt
python poi_studio.py
```

Training defaults: 150 epochs, imgsz 960, batch 4, lr 0.002, early-stopping patience 30.

Large artifacts (datasets, crops, training runs, trained weights) are gitignored — the repo tracks the pipeline and the labeling manifest, not the data.

---

## Note on scope

This is a course project and a computer-vision exercise. It is **not** a medical device and is not validated for clinical or diagnostic use.
