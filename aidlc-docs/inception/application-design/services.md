# Services — Smart Storage

## Service: ApplicationService (main.py)

**Purpose**: Главный сервис-оркестратор, координирующий все компоненты.

**Orchestration Pattern**: Простой процедурный оркестратор (не микросервисы).

### Режим 1: Обработка видеопотока (основной)
```
1. Инициализация: Config → Pipeline → VideoProcessor → Visualizer → DataExporter
2. Цикл обработки:
   a. VideoProcessor.get_frame() → кадр
   b. Pipeline.run(кадр) → PipelineResult
   c. Visualizer.show_pipeline(result) → отображение на экране
   d. По нажатию клавиши 's': DataExporter.export(result.decision)
   e. По нажатию клавиши 'q': выход из цикла
3. Завершение: VideoProcessor.stop()
```

### Режим 2: Обработка одиночного изображения
```
1. Инициализация: Config → Pipeline → Visualizer → DataExporter
2. Загрузка изображения с диска
3. Pipeline.run(image) → PipelineResult
4. Visualizer.show_pipeline(result)
5. DataExporter.export(result.decision)
6. Ожидание нажатия клавиши для закрытия
```

### Управление клавишами
| Клавиша | Действие |
|---------|----------|
| `q` | Выход из приложения |
| `s` | Сохранить текущий результат в CSV |
| `c` | Захватить и обработать текущий кадр (в режиме видео) |
| `p` | Пауза/возобновление видеопотока |

## Service: PipelineService (внутри Pipeline)

**Purpose**: Координация последовательности этапов обработки.

```
enhance(image) → enhanced_image
    ↓
segment(enhanced_image) → mask
    ↓
clean(mask) → cleaned_mask
    ↓
detect(enhanced_image, cleaned_mask) → DetectionResult
    ↓
decide(detection) → Decision
    ↓
PipelineResult (все промежуточные + финальный результат)
```

## Service: ColorDetectionService (внутри ColorDetector)

**Purpose**: Координация двух методов детекции цвета.

```
detect_hsv(image, mask) → ColorResult (hsv)
detect_kmeans(image, mask) → ColorResult (kmeans)
    ↓
Сравнение результатов → выбор наиболее уверенного
    ↓
Итоговый ColorResult для DetectionResult
```
