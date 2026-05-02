# Code Generation Plan — Smart Storage

## Project Structure
- **Workspace Root**: `smart-storage/`
- **Application Code**: `smart-storage/src/`
- **Tests**: `smart-storage/tests/`
- **Test Images**: `smart-storage/test_images/`
- **Output**: `smart-storage/output/`

---

## Step 1: Project Setup (uv)
- [x] 1.1 Initialize uv project with `pyproject.toml`
- [x] 1.2 Create `src/__init__.py`
- [x] 1.3 Create `tests/__init__.py`
- [x] 1.4 Create `output/.gitkeep`
- [x] 1.5 Create `test_images/.gitkeep`

## Step 2: Data Models
- [x] 2.1 Create `src/models.py` — ColorResult, DetectionResult, Decision, PipelineResult dataclasses

## Step 3: Configuration
- [x] 3.1 Create `src/config.py` — AppConfig with all parameters, HSV ranges, classification rules dictionary

## Step 4: Color Detector
- [x] 4.1 Create `src/color_detector.py` — ColorDetector class
  - [x] 4.1.1 `detect_hsv()` — HSV color analysis
  - [x] 4.1.2 `detect_kmeans()` — K-means clustering color detection
  - [x] 4.1.3 `detect()` — run both methods, compare, return combined result
- [x] 4.2 Create `tests/test_color_detector.py` — unit tests for ColorDetector

## Step 5: Pipeline
- [x] 5.1 Create `src/pipeline.py` — Pipeline class
  - [x] 5.1.1 `enhance()` — CLAHE + Gaussian blur
  - [x] 5.1.2 `segment()` — Adaptive threshold segmentation
  - [x] 5.1.3 `clean()` — Morphological operations (CLOSE + OPEN + largest contour)
  - [x] 5.1.4 `detect()` — Contour analysis + color detection + size categorization
  - [x] 5.1.5 `decide()` — Rule-based classification with confidence scoring
  - [x] 5.1.6 `run()` — Full pipeline execution returning PipelineResult
- [x] 5.2 Create `tests/test_pipeline.py` — unit tests for Pipeline

## Step 6: Visualizer
- [x] 6.1 Create `src/visualizer.py` — Visualizer class
  - [x] 6.1.1 `create_dashboard()` — compose all pipeline stages into single image
  - [x] 6.1.2 `draw_detection()` — draw bounding box and labels
  - [x] 6.1.3 `show_pipeline()` — display dashboard via cv2.imshow

## Step 7: Video Processor
- [x] 7.1 Create `src/video_processor.py` — VideoProcessor class
  - [x] 7.1.1 `start()` / `stop()` — camera lifecycle
  - [x] 7.1.2 `get_frame()` — capture single frame
  - [x] 7.1.3 `is_running()` — status check

## Step 8: Data Exporter
- [x] 8.1 Create `src/data_exporter.py` — DataExporter class
  - [x] 8.1.1 `export()` — write single Decision to CSV
  - [x] 8.1.2 `export_batch()` — write multiple Decisions
- [x] 8.2 Create `tests/test_data_exporter.py` — unit tests for DataExporter

## Step 9: Main Application
- [x] 9.1 Create `src/main.py` — Application entry point
  - [x] 9.1.1 Video mode: webcam loop with Pipeline + Visualizer + keyboard controls
  - [x] 9.1.2 Image mode: single file processing
  - [x] 9.1.3 CLI argument parsing (--mode, --source, --output)
  - [x] 9.1.4 Error handling (no camera, invalid file, etc.)

## Step 10: Documentation
- [x] 10.1 Create `README.md` — installation, usage, examples
- [x] 10.2 Create `aidlc-docs/construction/smart-storage/code/code-summary.md` — code generation summary

---

## Story Traceability
| Step | Requirements Covered |
|------|---------------------|
| Step 2 | Data models for all pipeline stages |
| Step 3 | FR-06 (classification rules), NFR-01 (config) |
| Step 4 | FR-05 (color detection, two methods) |
| Step 5 | FR-02 (enhance), FR-03 (segment), FR-04 (clean), FR-05 (detect), FR-06 (decide) |
| Step 6 | FR-07 (visualization), FR-10 (pipeline output) |
| Step 7 | FR-01 (image acquisition), FR-08 (video processing) |
| Step 8 | FR-09 (data export) |
| Step 9 | FR-01, FR-07, FR-08 (application coordination) |
| Step 10 | NFR-03 (reproducibility), NFR-05 (documentation) |
