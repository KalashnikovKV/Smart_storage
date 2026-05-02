# Component Dependencies — Smart Storage

## Dependency Matrix

| Component | Depends On | Used By |
|-----------|-----------|---------|
| Config | — (standalone) | Pipeline, ColorDetector, Application |
| ColorDetector | Config, numpy, sklearn | Pipeline |
| Pipeline | Config, ColorDetector, numpy, cv2 | Application |
| Visualizer | numpy, cv2 | Application |
| VideoProcessor | cv2 | Application |
| DataExporter | pandas | Application |
| Application | Pipeline, Visualizer, VideoProcessor, DataExporter, Config | — (entry point) |

## Communication Patterns

### Data Flow

```
+------------------+     +----------+     +------------+
| VideoProcessor   |---->| Pipeline |---->| Visualizer |
| (camera frames)  |     | (CV proc)|     | (display)  |
+------------------+     +----------+     +------------+
                              |
                              |  PipelineResult
                              v
                        +--------------+
                        | DataExporter |
                        | (CSV output) |
                        +--------------+
```

### Component Interaction

```
+-------------+
| Application |
+------+------+
       |
       | coordinates
       |
+------+------+------+------+------+
|      |      |      |      |      |
v      v      v      v      v      v
Config Video  Pipe   Vis    Data   Color
       Proc   line   ualiz  Export Detect
              |                    ^
              | uses               |
              +--------------------+
```

## External Dependencies

| Library | Version | Used By | Purpose |
|---------|---------|---------|---------|
| opencv-python | 4.9.x | Pipeline, Visualizer, VideoProcessor | CV operations, GUI, camera |
| numpy | 1.26.x | Pipeline, ColorDetector, Visualizer | Array operations |
| scikit-learn | 1.4.x | ColorDetector | K-means clustering |
| pandas | 2.2.x | DataExporter | CSV export |

## File Structure

```
smart-storage/
+-- src/
|   +-- __init__.py
|   +-- main.py           # Application entry point
|   +-- pipeline.py        # Pipeline class
|   +-- color_detector.py  # ColorDetector class
|   +-- visualizer.py      # Visualizer class
|   +-- video_processor.py # VideoProcessor class
|   +-- data_exporter.py   # DataExporter class
|   +-- config.py          # Configuration and classification rules
|   +-- models.py          # Data models (dataclasses)
+-- test_images/           # Sample test images
+-- output/                # Output directory for results
+-- requirements.txt       # Python dependencies
+-- README.md              # Installation and usage instructions
```
