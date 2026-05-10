# Smart Storage — IT Peripheral Recognition

Computer Vision pipeline for automatic classification of IT peripherals by color and size.

```
Image → Enhance → Segment → Clean → Detect → Decision
```

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

Install only what you need — groups are independent of each other:

| Group | What it installs | Command |
|-------|-----------------|---------|
| `ml-tabular` | LightGBM, joblib, matplotlib, albumentations | `uv sync --group ml-tabular` |
| `ml-cnn` | PyTorch, torchvision, timm, albumentations, **Ultralytics YOLO** | `uv sync --group ml-cnn` |
| `notebook` | Jupyter, matplotlib, seaborn | `uv sync --group notebook` |
| `dev` | pytest, httpx | `uv sync --group dev` |

Multiple groups can be combined:

```bash
uv sync --group ml-cnn --group notebook
```

> **Note:** `uv sync` without `--group` installs only core dependencies and does **not** pull in torch or any ML libraries.

## Usage

### Webcam mode (real-time)

```bash
uv run python -m src.main --mode video
```

Place one object on a plain white or black background in front of the camera.  
The dashboard shows all 6 pipeline stages simultaneously.

**Controls:**

| Key | Action |
|-----|--------|
| `s` | Save current result to CSV |
| `c` | Freeze current frame |
| `p` | Pause / resume |
| `q` | Quit |

### Image mode (single file)

```bash
uv run python -m src.main --mode image --source test_images/Image.jpeg
```

Result is printed to console and saved to `output/results.csv` automatically.

### Custom CSV output path

```bash
uv run python -m src.main --mode image --source photo.jpg --output my_results.csv
```

## Tips for best results

- Use a **plain white or black background** — the segmentation works best on solid backgrounds
- Place **one object at a time** in the frame
- Ensure **good lighting** — avoid shadows on the object
- Dark objects → white background; light objects → dark background

## Output

Each run appends a row to `output/results.csv`:

| Field | Description |
|-------|-------------|
| timestamp | ISO 8601 datetime |
| category | Detected peripheral type |
| color | Dominant color |
| size_category | small / medium / large / long_thin |
| confidence | 0.0 – 1.0 |
| method | hsv / kmeans / combined |
| color_hsv | HSV method color name |
| color_kmeans | K-means method color name |

## Categories

| Category | Color | Size |
|----------|-------|------|
| iPhone Charger | white | small |
| Android Charger | black | small |
| USB-C Cable | white | long_thin |
| Power Cable | black | long_thin |
| Mouse | black / blue | medium |
| Keyboard | black / gray | large |
| Flash Drive | gray / silver | small |
| MacBook Charger | white | medium (compact) |

## Run tests

```bash
uv run pytest tests/ -v
```
