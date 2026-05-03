# Business Rules — Smart Storage

## BR-01: Size Categorization

### Size Categories (4 categories)

Size is determined by two factors: **area in pixels** (relative to frame) and **aspect ratio**.

```
area_ratio = object_area / frame_area

IF aspect_ratio > 4.0:
    size = "long_thin"    # Cables, long connectors
ELIF area_ratio < 0.02:
    size = "small"        # Chargers, flash drives, small adapters
ELIF area_ratio < 0.08:
    size = "medium"       # Mouse, small peripherals
ELSE:
    size = "large"        # Keyboard, large devices
```

### Threshold Table
| Category | Area Ratio | Aspect Ratio | Examples |
|----------|-----------|--------------|----------|
| small | < 2% of frame | < 4.0 | Зарядка, флешка |
| medium | 2% - 8% of frame | < 4.0 | Мышь |
| large | > 8% of frame | < 4.0 | Клавиатура |
| long_thin | any | > 4.0 | Кабели |

---

## BR-02: Classification Rules (Color + Size -> Category)

### Primary Rules Table
| Rule ID | Color | Size | Category (RU) | Category (EN) | Base Confidence |
|---------|-------|------|---------------|---------------|-----------------|
| R01 | white | small | Зарядка iPhone | iPhone Charger | 0.85 |
| R02 | black | small | Зарядка Android | Android Charger | 0.80 |
| R03 | black | long_thin | Кабель питания | Power Cable | 0.85 |
| R04 | white | long_thin | Кабель USB-C | USB-C Cable | 0.80 |
| R05 | black | medium | Мышь | Mouse | 0.85 |
| R06 | white | medium | Мышь (белая) | Mouse (White) | 0.80 |
| R07 | black | large | Клавиатура | Keyboard | 0.90 |
| R08 | white | large | Клавиатура (белая) | Keyboard (White) | 0.85 |
| R09 | gray | small | Флешка | Flash Drive | 0.80 |
| R10 | silver | small | Флешка | Flash Drive | 0.80 |

### Rule Matching Algorithm
```
1. Find all rules where color matches detected color
2. Among matches, find rules where size matches detected size
3. IF exact match (color + size):
     confidence = base_confidence * color_confidence
4. IF partial match (only color OR only size):
     confidence = base_confidence * 0.5 * match_confidence
5. IF no match:
     category = "Неизвестный объект"
     closest = rule with minimum distance
     confidence = 0.3
```

---

## BR-03: Confidence Calculation

### Final Confidence Formula
```
final_confidence = base_confidence * color_confidence * size_match_factor

WHERE:
  base_confidence = from classification rules table (0.80 - 0.90)
  color_confidence = from ColorDetector (0.0 - 1.0)
  size_match_factor:
    1.0 = exact size match
    0.7 = adjacent size category (e.g., small vs medium)
    0.3 = distant size category (e.g., small vs large)
```

### Confidence Thresholds
| Confidence Range | Display |
|-----------------|---------|
| >= 0.70 | "Категория — XX%" (high confidence) |
| 0.40 - 0.69 | "Категория? — XX%" (uncertain) |
| < 0.40 | "Неизвестный объект (ближайшее: Категория — XX%)" |

---

## BR-04: Unknown Object Handling

```
IF final_confidence < 0.40:
    display = "Неизвестный объект"
    additional_info = "Ближайшее: {closest_category} — {confidence}%"
    additional_info += "Цвет: {detected_color}, Размер: {detected_size}"
    
    Decision:
      category = "Unknown ({closest_category})"
      confidence = final_confidence
```

---

## BR-05: Error Handling Rules

| Scenario | Behavior |
|----------|----------|
| No object detected (empty mask) | Display "Объект не найден" |
| Object too small (< 0.5% of frame) | Display "Объект слишком мал" |
| Object too large (> 50% of frame) | Display "Объект слишком велик — проверьте расстояние" |
| Camera not available | Display error, offer file input mode |
| Multiple contours after cleaning | Use largest contour only |
| Color detection fails | Use "unknown" color, reduce confidence by 50% |

---

## BR-06: CSV Export Format

### Fields
```
timestamp, category, color, size_category, confidence, method, 
area_pixels, aspect_ratio, color_hsv, color_kmeans, 
confidence_hsv, confidence_kmeans
```

### Example Row
```
2026-04-24T12:30:00, Зарядка iPhone, white, small, 0.87, combined,
15234, 1.3, white, white, 0.91, 0.85
```
