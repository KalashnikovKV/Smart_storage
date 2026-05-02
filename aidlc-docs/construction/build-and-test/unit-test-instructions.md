# Unit Test Instructions — Smart Storage

## Run All Unit Tests
```bash
uv run pytest tests/test_data_exporter.py tests/test_color_detector.py tests/test_pipeline.py -v
```

## Run Individual Test Files
```bash
# Color detection tests
uv run pytest tests/test_color_detector.py -v

# Pipeline tests
uv run pytest tests/test_pipeline.py -v

# Data export tests
uv run pytest tests/test_data_exporter.py -v
```

## Expected Results
```
15 passed in ~3s
```

## Test Coverage

| Test File | Tests | What's Covered |
|-----------|-------|----------------|
| `test_color_detector.py` | 6 | HSV white/black, K-means white/black, combined agreement, empty mask |
| `test_pipeline.py` | 7 | enhance shape, segment binary, clean binary, full pipeline (white/black), empty image, content preservation |
| `test_data_exporter.py` | 2 | CSV creation, row appending |

## Notes on Warnings
K-means tests may show `ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (3)`.
This is **expected** for synthetic test images with uniform color. On real camera images this will not occur.
