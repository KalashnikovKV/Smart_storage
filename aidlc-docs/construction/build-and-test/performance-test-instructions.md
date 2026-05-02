# Performance Test Instructions — Smart Storage

## Performance Requirements (from NFR-02)
- Single frame processing: < 500ms
- Video mode: minimum 5-10 FPS
- Application startup: < 5 seconds

## Test: Single Frame Processing Time

### Run
```bash
uv run python -c "
import cv2, numpy as np, time
from src.pipeline import Pipeline

pipeline = Pipeline()
# Create a test image
image = np.zeros((480, 640, 3), dtype=np.uint8)
image[200:260, 280:340] = [255, 255, 255]

# Warm up
pipeline.run(image)

# Measure 10 runs
times = []
for _ in range(10):
    start = time.time()
    pipeline.run(image)
    times.append((time.time() - start) * 1000)

print(f'Avg: {sum(times)/len(times):.1f}ms')
print(f'Max: {max(times):.1f}ms')
print(f'Min: {min(times):.1f}ms')
print(f'Target: < 500ms')
print(f'Status: {\"PASS\" if max(times) < 500 else \"FAIL\"}')
"
```

### Expected Output
```
Avg: ~50-150ms
Max: < 500ms
Status: PASS
```

## Test: Video FPS (requires webcam)
The `processing_time_ms` field in `PipelineResult` tracks per-frame time.
Monitor it in the dashboard — it's displayed in the Decision panel.

Target: `processing_time_ms < 200ms` per frame → ~5+ FPS achievable.
