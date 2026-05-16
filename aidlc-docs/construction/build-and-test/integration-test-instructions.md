# Integration Test Instructions — Smart Storage

## Purpose
Test the full pipeline end-to-end with real images to verify all components work together.

## Scenario 1: Full Pipeline on Test Image

### Setup
Place a test image in `training/test_images/` (e.g., a photo of a white iPhone charger on a black background).

### Test Steps
```bash
uv run python -m src.main --mode image --source training/test_images/charger_white.jpg --output output/test_results.csv
```

### Expected Results
- Dashboard window opens showing all 6 pipeline stages
- Console output shows detected category, confidence, color, size
- `output/test_results.csv` is created with one row

---

## Scenario 2: Pipeline → CSV Export Integration

### Test Steps
```bash
# Process multiple images
uv run python -m src.main --mode image --source training/test_images/mouse_black.jpg --output output/test_results.csv
uv run python -m src.main --mode image --source training/test_images/cable_black.jpg --output output/test_results.csv
```

### Expected Results
- `output/test_results.csv` contains 2 rows (appended, not overwritten)
- Each row has: timestamp, category, color, size_category, confidence, method, area_pixels, aspect_ratio, color_hsv, color_kmeans, confidence_hsv, confidence_kmeans

---

## Scenario 3: Video Mode Integration (requires webcam)

### Test Steps
```bash
uv run python -m src.main --mode video
```

### Expected Results
- Dashboard window opens with live video
- Pipeline processes each frame in real time
- Press `s` to save a result → CSV row appended
- Press `q` to quit cleanly

---

## Recommended Test Images
Create or photograph these items on a solid white/black background:

| File | Object | Expected Category |
|------|--------|------------------|
| `charger_white.jpg` | White iPhone charger | Зарядка iPhone |
| `charger_black.jpg` | Black Android charger | Зарядка Android |
| `cable_black_long.jpg` | Black power cable | Кабель питания |
| `mouse_black.jpg` | Black mouse | Мышь |
| `flash_drive_gray.jpg` | Gray flash drive | Флешка |
