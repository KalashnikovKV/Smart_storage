# Application Design — Smart Storage: IT Peripheral Recognition System

## Overview

Smart Storage — система компьютерного зрения для автоматического распознавания IT-периферии (мышки, зарядки, кабели и т.д.) по цвету и размеру. Реализует полный CV-пайплайн: image → enhance → segment → clean → detect → decision.

**Архитектурные решения:**
- GUI: OpenCV highgui (cv2.imshow) — минимальный подход
- Организация пайплайна: один класс `Pipeline` с методами для каждого этапа
- Конфигурация: Python-словарь в модуле `config.py`

---

## Components (7 компонентов)

### 1. Pipeline
Основной класс CV-пайплайна. Методы: `enhance()`, `segment()`, `clean()`, `detect()`, `decide()`, `run()`. Принимает изображение, возвращает `PipelineResult` со всеми промежуточными результатами.

### 2. ColorDetector
Детекция цвета двумя методами: HSV-анализ и K-means кластеризация. Используется внутри Pipeline на этапе detect.

### 3. Config
Конфигурация системы: правила классификации (цвет+размер → категория), параметры пайплайна, HSV-диапазоны цветов, пороги размеров.

### 4. Visualizer
Визуализация через OpenCV highgui. Создаёт дашборд со всеми этапами пайплайна, рисует bounding box и метки.

### 5. VideoProcessor
Захват видеопотока с веб-камеры. Управление циклом захвата (старт/стоп/пауза).

### 6. DataExporter
Экспорт результатов в CSV. Поля: timestamp, категория, цвет, размер, уверенность, метод.

### 7. Application (main.py)
Точка входа. Координирует все компоненты. Два режима: видеопоток и одиночное изображение.

---

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
    method_used: str

@dataclass
class PipelineResult:
    original: np.ndarray
    enhanced: np.ndarray
    mask: np.ndarray
    cleaned_mask: np.ndarray
    detection: DetectionResult
    decision: Decision
```

---

## Pipeline Flow

```
Input Image
    |
    v
enhance() --> Enhanced Image
    |
    v
segment() --> Segmentation Mask
    |
    v
clean() --> Cleaned Mask
    |
    v
detect() --> DetectionResult (bbox, color, size)
    |           |
    |           +--> ColorDetector.detect_hsv()
    |           +--> ColorDetector.detect_kmeans()
    |
    v
decide() --> Decision (category + confidence)
    |
    v
PipelineResult (all intermediate + final)
```

---

## Component Dependencies

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

---

## File Structure

```
smart-storage/
+-- src/
|   +-- __init__.py
|   +-- main.py
|   +-- pipeline.py
|   +-- color_detector.py
|   +-- visualizer.py
|   +-- video_processor.py
|   +-- data_exporter.py
|   +-- config.py
|   +-- models.py
+-- test_images/
+-- output/
+-- requirements.txt
+-- README.md
```

---

## User Interaction

| Клавиша | Действие |
|---------|----------|
| `q` | Выход |
| `s` | Сохранить результат в CSV |
| `c` | Захватить кадр |
| `p` | Пауза/возобновление |

---

## External Dependencies

| Library | Purpose |
|---------|---------|
| opencv-python 4.9.x | CV operations, GUI, camera |
| numpy 1.26.x | Array operations |
| scikit-learn 1.4.x | K-means clustering |
| pandas 2.2.x | CSV export |
