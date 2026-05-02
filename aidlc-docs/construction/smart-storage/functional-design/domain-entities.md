# Domain Entities — Smart Storage

## Entity: ColorResult
```
ColorResult:
  name: str           # Color name: "white", "black", "gray", "silver", "red", "blue"
  confidence: float   # 0.0 - 1.0
  method: str         # "hsv" or "kmeans"
  rgb: tuple          # (R, G, B) dominant color values
```

## Entity: DetectionResult
```
DetectionResult:
  bbox: tuple              # (x, y, w, h) bounding rectangle
  color_hsv: ColorResult   # HSV method result
  color_kmeans: ColorResult # K-means method result
  primary_color: str       # Final determined color name
  size_category: str       # "small", "medium", "large", "long_thin"
  area_pixels: int         # Object area in pixels
  aspect_ratio: float      # max(w,h) / min(w,h)
  contour: ndarray         # OpenCV contour points
```

## Entity: Decision
```
Decision:
  category: str       # "Зарядка iPhone", "Кабель питания", "Unknown (Mouse)"
  confidence: float   # 0.0 - 1.0
  color: str          # Detected color name
  size: str           # Detected size category
  method_used: str    # "hsv", "kmeans", "combined"
  is_unknown: bool    # True if confidence < 0.40
  closest_match: str  # Closest category if unknown
  timestamp: str      # ISO 8601 timestamp
```

## Entity: PipelineResult
```
PipelineResult:
  original: ndarray        # Original input image
  enhanced: ndarray        # After enhancement
  mask: ndarray            # After segmentation
  cleaned_mask: ndarray    # After cleaning
  detection: DetectionResult  # Detection results
  decision: Decision       # Final classification
  processing_time_ms: float   # Total pipeline time
```

## Entity: ClassificationRule
```
ClassificationRule:
  rule_id: str        # "R01", "R02", etc.
  color: str          # Expected color
  size: str           # Expected size category
  category_ru: str    # Category name in Russian
  category_en: str    # Category name in English
  base_confidence: float  # 0.80 - 0.90
```

## Entity: AppConfig
```
AppConfig:
  # Enhancement
  clahe_clip_limit: float       # default: 2.0
  clahe_tile_size: tuple        # default: (8, 8)
  blur_kernel: tuple            # default: (5, 5)
  
  # Segmentation
  adaptive_block_size: int      # default: 11
  adaptive_c: int               # default: 2
  pre_blur_kernel: tuple        # default: (11, 11)
  
  # Cleaning
  close_kernel_size: tuple      # default: (7, 7)
  close_iterations: int         # default: 2
  open_kernel_size: tuple       # default: (5, 5)
  open_iterations: int          # default: 2
  
  # Size thresholds
  small_max_ratio: float        # default: 0.02
  medium_max_ratio: float       # default: 0.08
  long_thin_aspect: float       # default: 4.0
  min_object_ratio: float       # default: 0.005
  max_object_ratio: float       # default: 0.50
  
  # K-means
  kmeans_clusters: int          # default: 3
  kmeans_max_iter: int          # default: 100
  
  # HSV ranges
  hsv_ranges: dict              # Color name -> HSV range mapping
  
  # Classification rules
  rules: list[ClassificationRule]
  
  # Confidence thresholds
  high_confidence: float        # default: 0.70
  low_confidence: float         # default: 0.40
  
  # Camera
  camera_id: int                # default: 0
  
  # Export
  csv_output_path: str          # default: "output/results.csv"
```

## Data Flow Diagram

```
Camera/File
    |
    v
[np.ndarray BGR]
    |
    v
enhance() --> [np.ndarray BGR enhanced]
    |
    v
segment() --> [np.ndarray binary mask]
    |
    v
clean() --> [np.ndarray clean mask]
    |
    v
detect() --> [DetectionResult]
    |              |
    |    +---------+---------+
    |    |                   |
    |    v                   v
    | HSV analysis     K-means analysis
    |    |                   |
    |    v                   v
    | [ColorResult]    [ColorResult]
    |    |                   |
    |    +---------+---------+
    |              |
    v              v
decide() --> [Decision]
    |
    v
[PipelineResult] --> Visualizer (display)
                 --> DataExporter (CSV)
```
