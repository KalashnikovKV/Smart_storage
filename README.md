# Smart Storage — IT Peripheral Recognition

Computer Vision pipeline for automatic classification of IT peripherals by color and size.

```
Image → Enhance → Segment → Clean → Detect → Decision
```

Two modes of operation:
- **Rule-based** — works out of the box, no model needed
- **YOLO-Seg** — uses a trained YOLO-Seg model for accurate instance segmentation

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Installation

```bash
git clone <repo-url>
cd smart-storage
uv sync
```

### Optional dependency groups

| Group | What it installs | Command |
|-------|-----------------|---------|
| `yolo` | Ultralytics YOLO (for YOLO-Seg mode) | `uv sync --group yolo` |
| `notebook` | Jupyter, matplotlib, seaborn | `uv sync --group notebook` |
| `dev` | pytest | `uv sync --group dev` |

> `uv sync` without `--group` installs only core dependencies (OpenCV, NumPy, scikit-learn).

### Local data (not in git)

Photos, videos, pipeline outputs, and trained weights stay on your machine:

| Path | Put here |
|------|----------|
| `training/test_images/` | Your `.jpg` / `.png` for batch, label, inference |
| `training/test_video/` | Your `.mov` / `.mp4` for `--mode label --source …` |
| `output/` | `results.csv` (predictions + `ground_truth`) in git; ROI and `stages/` masks stay local |
| `dataset/` | Built by `training/build_dataset.py` for YOLO training |
| `models/` | Downloaded or trained `.pt` weights (e.g. `yolo11n-seg.pt`) |

After clone, copy your files into `training/test_images/` (folders are empty except `.gitkeep`).

---

## Project structure

```
src/
├── pipeline/     # 5 CV stages (enhance → segment → clean → detect → decide)
├── ml/           # YOLO-Seg backend (--model)
├── modes/        # CLI: video, image, batch, label
└── app/          # GUI, CSV export, labeling
scripts/          # build_dataset.py, train_yolo.py wrappers
training/         # dataset builder, YOLO trainer, test media
```

---

## Usage

### Quick start — rule-based mode (no model needed)

```bash
# Webcam
./run.sh --mode video

# Single image — rule-based demo
./run.sh --mode image --source training/test_images/Image_6.jpeg --no-display

# Single image — YOLO demo
./run.sh --mode image --source training/test_images/Image_2.jpeg --model models/yolo11n-seg.pt --device cpu --no-display

# Batch folder (predictions + masks for YOLO dataset)
./run.sh --mode batch --source training/test_images/

# Manual labeling (ground_truth) — disputed images only
./run.sh --mode label --source training/test_images/Image.jpeg
```

`image` = quick preview (window, no console prompt). `label` = confirm/override class → `ground_truth` in `output/results.csv`.

Or without the script:

```bash
uv run python -m src.main --mode image --source training/test_images/Image.jpeg
```

---

### YOLO-Seg mode (requires trained model)

First install YOLO dependencies:

```bash
uv sync --group yolo
```

Then run with `--model` and optionally `--device`:

```bash
# Webcam — CPU
uv run python -m src.main --mode video --model models/yolo11n-seg.pt --device cpu

# Webcam — GPU (CUDA)
uv run python -m src.main --mode video --model models/yolo11n-seg.pt --device cuda

# Single image
uv run python -m src.main --mode image --source training/test_images/Image_1.jpeg --model models/yolo11n-seg.pt

# Batch
uv run python -m src.main --mode batch --source training/test_images/ --model models/yolo11n-seg.pt
```

In YOLO-Seg mode the pipeline uses:
- YOLO-Seg for object detection and segmentation (class + bbox + mask)
- OpenCV HSV/K-means for color detection **inside the YOLO mask**
- Mask area for size estimation

---

### All CLI arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `video`, `image`, `batch`, or `label` | `video` |
| `--source` | Image path (image mode) or folder path (batch mode) | — |
| `--model` | Path to YOLO-Seg weights `.pt` | None (rule-based) |
| `--device` | Inference device: `cpu`, `cuda`, `cuda:0`, `mps` | auto-detect |
| `--output` | CSV output path | `output/results.csv` |
| `--no-save-images` | Do not save pipeline stage images | off |
| `--no-display` | Run without OpenCV window (headless) | off |
| `--debug` | Enable debug logging | off |

> When `--device` is omitted, CUDA is auto-detected and falls back to `cpu` with a warning if unavailable.

---

### Controls (video mode)

| Key / Action | Effect |
|-----|--------|
| `q` | Quit |
| Close window (×) | Quit |
| `s` | Save current result to CSV |
| `c` | Freeze current frame |
| `p` | Pause / resume |
| `space` | Pause / resume (label mode with video file or webcam) |

---

## Output

Each run appends a row to `output/results.csv`:

| Field | Description |
|-------|-------------|
| timestamp | ISO 8601 datetime |
| category | Detected peripheral type |
| color | Dominant color |
| size_category | small / medium / large / long_thin |
| confidence | 0.0 – 1.0 |
| method_used | `yolo` / `combined` / `hsv` / `kmeans` |
| color_hsv | HSV method color name |
| color_kmeans | K-means method color name |

Pipeline stage images are saved to `output/stages/` on each run.

---

## Categories

| Category | Color | Size |
|----------|-------|------|
| Mouse | black / gray / blue | medium |
| Keyboard | black / gray / silver | large |
| Charger Adapter | white / silver / gray | small / medium |
| USB-C Cable | white / gray / black | long_thin / medium |
| Headphones | black / white / silver | medium / large |
| Flash Drive | gray / silver / black | small |
| Colored Object | any chromatic color | any |

---

## Run tests

```bash
uv run pytest tests/ -v
```

---

## Training a YOLO-Seg model

See [aidlc-docs/ml-roadmap.md](aidlc-docs/ml-roadmap.md) for the full ML roadmap.

Quick start:

```bash
# 1. Batch — predictions + cleaned_mask artifacts
./run.sh --mode batch --source training/test_images/

# 2. Label — only for images you want to correct (writes ground_truth to output/results.csv)
./run.sh --mode label --source training/test_images/Image.jpeg

# 3. Build YOLO-Seg dataset (reads output/results.csv + output/stages/)
uv run python scripts/build_dataset.py --clean

# Or import Roboflow/CVAT YOLOv8 Segmentation export:
uv run python scripts/build_dataset.py --import-from path/to/export --clean

# 4. Train
uv run python scripts/train_yolo.py \
    --data dataset/dataset.yaml \
    --model yolo11n-seg.pt \
    --epochs 100

# 5. Run with trained model
./run.sh --mode image --source training/test_images/Image.jpeg --model runs/segment/smart_storage/weights/best.pt
```

For precise polygon masks use Roboflow or CVAT (YOLOv8 Segmentation export) instead of auto masks from `cleaned_mask`.
