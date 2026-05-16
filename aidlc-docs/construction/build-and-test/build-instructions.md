# Build Instructions — Smart Storage

## Prerequisites
- **Python**: 3.12+
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (`pip install uv` or `brew install uv`)
- **Webcam**: Required for video mode (optional for image/batch mode)
- **OS**: Linux / macOS / Windows
- **Disk Space**: ~500 MB (venv + OpenCV); ~2 GB with YOLO/torch

## Build Steps

### 1. Clone and enter project
```bash
git clone <repo-url>
cd smart-storage
```

### 2. Install core dependencies
```bash
uv sync
```
Creates `.venv/` and installs all pinned dependencies from `uv.lock`.

### 3. (Optional) Install YOLO dependencies
Required only when using `--model`:
```bash
uv sync --group yolo
```

### 4. Verify installation
```bash
uv run python -c "import cv2, numpy, sklearn, pandas; print('Core OK')"
# With YOLO:
uv run python -c "from ultralytics import YOLO; import torch; print('YOLO OK, torch', torch.__version__)"
```

### 5. Place model weights
Put your `.pt` file in the `models/` folder:
```bash
mkdir -p models
# copy your weights, e.g.:
cp ~/Downloads/yolo11n-seg.pt models/
```

---

## Running the application

### Rule-based mode (no model required)
```bash
# Webcam
uv run python -m src.main --mode video

# Single image
uv run python -m src.main --mode image --source training/test_images/Image_1.jpeg

# Batch folder
uv run python -m src.main --mode batch --source training/test_images/
```

### YOLO-Seg mode
```bash
# Webcam — CPU (always works)
uv run python -m src.main --mode video --model models/yolo11n-seg.pt --device cpu

# Webcam — GPU auto-detect (uses CUDA if available, falls back to cpu)
uv run python -m src.main --mode video --model models/yolo11n-seg.pt

# Webcam — explicit GPU
uv run python -m src.main --mode video --model models/yolo11n-seg.pt --device cuda

# Single image
uv run python -m src.main --mode image --source training/test_images/Image_1.jpeg --model models/yolo11n-seg.pt

# Batch
uv run python -m src.main --mode batch --source training/test_images/ --model models/yolo11n-seg.pt
```

### All CLI arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `video`, `image`, or `batch` | `video` |
| `--source` | Image path (image mode) or folder path (batch mode) | — |
| `--model` | Path to YOLO-Seg weights `.pt` | None (rule-based) |
| `--device` | Inference device: `cpu`, `cuda`, `cuda:0`, `mps` | auto-detect |
| `--output` | CSV output path | `output/results.csv` |
| `--no-save-images` | Do not save pipeline stage images | off |
| `--no-display` | Run without OpenCV window (headless) | off |
| `--debug` | Enable debug logging | off |

> `--device` is only used when `--model` is provided. Without `--model` the built-in CV pipeline runs on CPU.

### Controls (video mode)

| Key / Action | Effect |
|---|---|
| `q` | Quit |
| Close window (×) | Quit |
| `s` | Save current result to CSV |
| `c` | Freeze current frame |
| `p` | Pause / resume |

---

## GPU / CUDA setup

### Check current status
```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('torch:', torch.__version__); print('cuda compiled:', torch.version.cuda)"
nvidia-smi
```

### PyTorch CUDA version mismatch
If `torch.cuda.is_available()` returns `False` despite having an NVIDIA GPU, the installed PyTorch may be compiled for a different CUDA version than your driver supports.

Check driver max CUDA version:
```bash
cat /proc/driver/nvidia/version   # shows driver version, e.g. 570.x → supports CUDA ≤ 12.8
```

Reinstall PyTorch matching your driver:

| Driver version | Max CUDA | Install command |
|---|---|---|
| 525–535 | 12.0 | `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |
| 545–560 | 12.4 | `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
| 565–575 | 12.6 | `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall` |
| 575+ | 12.8 | `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall` |

After reinstall verify:
```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Build Artifacts
- `.venv/` — Python virtual environment
- `uv.lock` — Locked dependency versions (reproducible builds)
- `output/` — CSV results + pipeline stage images
- `models/` — YOLO weights (not tracked by git)

## Installed Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.11.x | CV operations, GUI, camera |
| numpy | 1.26.x | Array operations |
| scikit-learn | 1.5.x | K-means clustering |
| pandas | 2.2.x | CSV export |
| ultralytics | 8.2+ | YOLO-Seg inference (optional) |
| torch | 2.x | YOLO backend (installed with ultralytics) |
| pytest | 8.4.x | Testing |

---

## Troubleshooting

### `uv: command not found`
```bash
pip install uv
# or on macOS:
brew install uv
```

### OpenCV: no display / `cannot connect to X server`
```bash
# Use headless mode:
uv run python -m src.main --mode image --source training/test_images/Image_1.jpeg --no-display
# Or replace opencv-python with opencv-python-headless in pyproject.toml
```

### Camera not found
```bash
# Test camera directly:
uv run python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened()); ret,f=cap.read(); print(f.shape if ret else 'FAILED')"
# Fallback to image mode:
uv run python -m src.main --mode image --source training/test_images/Image_1.jpeg
```

### NNPACK warnings on startup
```
[W] Could not initialize NNPACK! Reason: Unsupported hardware.
```
Harmless — PyTorch falls back to a different kernel. Suppress with:
```bash
uv run python -m src.main --mode video 2>/dev/null
```

### CUDA warning: driver too old
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
```
See **GPU / CUDA setup** section above — reinstall PyTorch for your driver version.
