# Component Methods — Smart Storage

## Pipeline

```python
class Pipeline:
    def __init__(self, config: dict)
    def run(self, image: np.ndarray) -> PipelineResult
    def enhance(self, image: np.ndarray) -> np.ndarray
    def segment(self, image: np.ndarray) -> np.ndarray
    def clean(self, mask: np.ndarray) -> np.ndarray
    def detect(self, image: np.ndarray, mask: np.ndarray) -> DetectionResult
    def decide(self, detection: DetectionResult) -> Decision
```

- `__init__`: Инициализация с конфигурацией (пороги, параметры)
- `run`: Выполнение полного пайплайна, возврат всех промежуточных результатов
- `enhance`: Улучшение изображения (контраст, шумоподавление)
- `segment`: Сегментация объекта от фона, возврат бинарной маски
- `clean`: Морфологическая очистка маски
- `detect`: Определение bounding box, цвета и размера объекта
- `decide`: Классификация объекта по правилам цвет+размер

## ColorDetector

```python
class ColorDetector:
    def __init__(self, config: dict)
    def detect_hsv(self, image: np.ndarray, mask: np.ndarray) -> ColorResult
    def detect_kmeans(self, image: np.ndarray, mask: np.ndarray) -> ColorResult
    def detect(self, image: np.ndarray, mask: np.ndarray) -> dict
```

- `detect_hsv`: Определение цвета через HSV-диапазоны
- `detect_kmeans`: Определение цвета через K-means кластеризацию
- `detect`: Запуск обоих методов, возврат сравнительного результата

## Visualizer

```python
class Visualizer:
    def __init__(self)
    def show_pipeline(self, result: PipelineResult) -> None
    def show_frame(self, title: str, image: np.ndarray) -> None
    def draw_detection(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray
    def create_dashboard(self, result: PipelineResult) -> np.ndarray
```

- `show_pipeline`: Отображение всех этапов пайплайна
- `show_frame`: Отображение одного кадра в окне OpenCV
- `draw_detection`: Рисование bounding box и меток на изображении
- `create_dashboard`: Компоновка всех этапов в единое изображение-дашборд

## VideoProcessor

```python
class VideoProcessor:
    def __init__(self, camera_id: int = 0)
    def start(self) -> None
    def stop(self) -> None
    def get_frame(self) -> Optional[np.ndarray]
    def is_running(self) -> bool
```

- `start`: Открытие видеопотока с камеры
- `stop`: Закрытие видеопотока и освобождение ресурсов
- `get_frame`: Захват одного кадра
- `is_running`: Проверка состояния потока

## DataExporter

```python
class DataExporter:
    def __init__(self, output_path: str = "results.csv")
    def export(self, decision: Decision) -> None
    def export_batch(self, decisions: list[Decision]) -> None
```

- `export`: Запись одного результата классификации в CSV
- `export_batch`: Запись нескольких результатов

## Data Models

```python
@dataclass
class ColorResult:
    name: str           # "white", "black", "silver"
    confidence: float   # 0.0 - 1.0
    method: str         # "hsv" or "kmeans"

@dataclass
class DetectionResult:
    bbox: tuple         # (x, y, w, h)
    color_hsv: ColorResult
    color_kmeans: ColorResult
    size_category: str  # "small", "medium", "large", "long_thin"
    area_pixels: int
    aspect_ratio: float

@dataclass
class Decision:
    category: str       # "Зарядка iPhone", "Кабель питания", etc.
    confidence: float   # 0.0 - 1.0
    color: str
    size: str
    method_used: str    # "hsv", "kmeans", or "combined"

@dataclass
class PipelineResult:
    original: np.ndarray
    enhanced: np.ndarray
    mask: np.ndarray
    cleaned_mask: np.ndarray
    detection: DetectionResult
    decision: Decision
```
