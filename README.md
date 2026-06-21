# Smart Storage — IT Peripheral Recognition

Computer Vision pipeline for automatic classification of IT peripherals by color and size.

```
Image → Enhance → Segment → Clean → Detect → Decision
```

Two inference modes:

- **Rule-based** — threshold segmentation + rules R01–R06 (no model file)
- **YOLO-Seg** — trained instance segmentation on your custom Roboflow dataset

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

---

## First-time setup

```bash
cd smart-storage
uv sync                  # core CV stack
uv sync --group yolo     # YOLO inference + training
uv sync --group dev      # pytest (optional)
```

Trained model (ready to use):

```
models/smart_storage_final_best.pt
```

Override paths/devices for wrapper scripts:

```bash
export SMART_STORAGE_MODEL=models/smart_storage_final_best.pt
export SMART_STORAGE_DEVICE=cuda    # or cpu
export SMART_STORAGE_EPOCHS=50
```

---

## Quick start (wrapper scripts)

All scripts run from the project root. Defaults use sample photos under `training/test_images/`.

| Script | What it does |
|--------|----------------|
| `./run_rule_image.sh [photo]` | Rule-based — one image, OpenCV dashboard |
| `./run_yolo_image.sh [photo]` | YOLO — one image, your `best.pt`, GPU |
| `./run_rule_batch.sh [folder]` | Rule-based — all images in folder |
| `./run_yolo_batch.sh [folder]` | YOLO — batch on folder |
| `./run_rule_video.sh [file]` | Rule-based — webcam or video file |
| `./run_yolo_video.sh [file]` | YOLO — webcam or video file |
| `./run_import_roboflow.sh <export-dir>` | Import Roboflow YOLOv8-Seg export → `dataset/` |
| `./run_validate_dataset.sh` | Check `dataset/` layout and labels |
| `./run_train_yolo.sh` | Train YOLO-Seg on `dataset/` |
| `./run_tests.sh` | Run pytest |
| `./run.sh …` | Low-level CLI (all flags) |

Examples:

```bash
./run_rule_image.sh training/test_images/Charger_Adapter/Image_43.jpg
./run_yolo_image.sh training/test_images/Charger_Adapter/Image_43.jpg
./run_yolo_batch.sh dataset/images/val
./run_yolo_video.sh training/test_video/Video_10.MOV
```

---

## Workflow: Roboflow → train → run

Annotation is done in **Roboflow** (app.roboflow.com), not in this repo.

```bash
# 1. Export YOLOv8 Segmentation from Roboflow, then import:
./run_import_roboflow.sh "path/to/roboflow-export"

# 2. Validate dataset/
./run_validate_dataset.sh

# 3. Train (starts from models/yolo11n-seg.pt backbone)
./run_train_yolo.sh

# 4. Inference with trained weights
./run_yolo_image.sh training/test_images/mouse/some_photo.jpg
```

Comparison metrics and report assets (rule vs YOLO) are already in:

- `output/comparison_val_summary.txt`
- `output/comparison_test_images_summary.txt`
- `docs/comparison-report/` (not modified by this README)

---

## Low-level CLI (`./run.sh`)

Same as `uv run python -m src.main`:

```bash
./run.sh --mode image --source training/test_images/Charger_Adapter/Image_43.jpg
./run.sh --mode image --source training/test_images/Charger_Adapter/Image_43.jpg \
  --model models/smart_storage_final_best.pt \
  --device cuda
./run.sh --mode batch --source training/test_images/mouse
./run.sh --mode video --source training/test_video/Video_10.MOV
```

| Flag | Description |
|------|-------------|
| `--mode` | `video`, `image`, `batch` |
| `--source` | Image path, folder, or video file |
| `--model` | Path to `.pt` weights (YOLO mode) |
| `--device` | `cpu`, `cuda` (YOLO only) |
| `--no-display` | No OpenCV window |
| `--output` | CSV path (default: `output/results.csv`) |

---

## Video controls

| Key | Webcam | Video file |
|-----|--------|------------|
| `q` | Quit | Quit |
| `s` | Save to CSV | Save to CSV |
| `c` / `p` | Capture / pause | — |
| `space` | — | Pause |
| `a` | — | Analyze paused frame |

---

## Project layout

```
src/
├── pipeline/     # 5 CV stages
├── ml/           # YOLO-Seg backend
├── modes/        # CLI: video, image, batch
└── app/          # OpenCV dashboard, CSV export
scripts/          # build_dataset.py, train_yolo.py wrappers
training/         # dataset builder + trainer implementation
dataset/          # YOLO images/labels (from Roboflow import)
training/test_images/   # local test photos by category
models/           # trained weights (smart_storage_final_best.pt + base backbone)
runs/             # raw training output (local, gitignored)
output/           # results.csv, stages/, comparison metrics
```

Test photos layout:

```
training/test_images/
├── Charger_Adapter/
├── Headphones/
├── Keyboard/
├── mouse/
├── USB-C_Cable/
└── flash_drive/
```

---

## Output

Each run appends rows to `output/results.csv`. Stage images go to `output/stages/`.

Evaluate rows that have `ground_truth` filled in:

```bash
uv run python -m src.evaluation --csv output/results.csv
```

---

## Tests

```bash
./run_tests.sh
```

---

## Categories

| Category | Typical color | Typical size |
|----------|---------------|--------------|
| Mouse | black / gray | medium |
| Keyboard | black / gray | large |
| Charger Adapter | white / silver | small / medium |
| USB-C Cable | white / gray / black | long_thin / medium |
| Headphones | black / white | medium / large |
| Flash Drive | gray / silver / black | small |

Known limitations: see `aidlc-docs/failure-cases.md`.
