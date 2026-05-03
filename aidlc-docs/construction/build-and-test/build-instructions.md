# Build Instructions — Smart Storage

## Prerequisites
- **Python**: 3.12+
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (install via `pip install uv` or `brew install uv`)
- **Webcam**: Required for video mode (optional for image mode)
- **OS**: macOS / Linux / Windows
- **Disk Space**: ~500MB (for Python venv + OpenCV)

## Build Steps

### 1. Clone and Enter Project
```bash
git clone <repo-url>
cd smart-storage
```

### 2. Install Dependencies
```bash
uv sync
```
This creates `.venv/` and installs all pinned dependencies automatically.

### 3. Verify Installation
```bash
uv run python -c "import cv2, numpy, sklearn, pandas; print('All dependencies OK')"
```

### 4. Create Output Directory
```bash
mkdir -p output
```

## Build Artifacts
- `.venv/` — Python virtual environment with all dependencies
- `uv.lock` — Locked dependency versions (reproducible builds)

## Installed Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.11.x | CV operations, GUI, camera |
| numpy | 1.26.x | Array operations |
| scikit-learn | 1.5.x | K-means clustering |
| pandas | 2.2.x | CSV export |
| pytest | 8.4.x | Testing |

## Troubleshooting

### `uv: command not found`
```bash
pip install uv
# or on macOS:
brew install uv
```

### OpenCV import error on Linux (no display)
```bash
pip install opencv-python-headless
# or add to pyproject.toml: opencv-python-headless instead of opencv-python
```

### Camera not found
- Check webcam is connected
- Try `--mode image --source test_images/sample.jpg` instead
