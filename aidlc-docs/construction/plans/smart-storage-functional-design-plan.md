# Functional Design Plan — Smart Storage

## Plan Steps

- [x] 1. Define detailed business logic for each pipeline stage
  - [x] 1.1 Enhancement algorithms and parameters
  - [x] 1.2 Segmentation algorithm and thresholds
  - [x] 1.3 Mask cleaning morphological operations
  - [x] 1.4 Detection logic (bounding box, size categorization)
- [x] 2. Define color detection business logic
  - [x] 2.1 HSV color ranges and mapping rules
  - [x] 2.2 K-means clustering parameters and color naming
  - [x] 2.3 Method comparison and confidence scoring
- [x] 3. Define classification decision rules
  - [x] 3.1 Size categories and thresholds
  - [x] 3.2 Color + Size → Category mapping rules
  - [x] 3.3 Confidence calculation algorithm
  - [x] 3.4 Fallback/unknown handling
- [x] 4. Define domain entities and data flow
- [x] 5. Define error handling and edge cases
- [x] 6. Generate business-logic-model.md
- [x] 7. Generate business-rules.md
- [x] 8. Generate domain-entities.md
- [x] 9. Validate design completeness

---

## Design Questions

Пожалуйста, ответьте на вопросы ниже.

### Question 1
Какой алгоритм сегментации предпочтителен для отделения предмета от однотонного фона?

A) Пороговая бинаризация (Otsu threshold) — простой и быстрый, хорошо работает на контрастном фоне
B) Adaptive threshold — адаптивный порог, лучше при неравномерном освещении
C) GrabCut — более точный, но медленнее
D) Other (please describe after [Answer]: tag below)

[B]: 

### Question 2
Какие размерные категории использовать для классификации?

A) 3 категории: маленький (small), средний (medium), большой (large) — по площади объекта
B) 4 категории: маленький, средний, большой, длинный/тонкий (long_thin) — с учётом aspect ratio
C) 5 категорий: очень маленький, маленький, средний, большой, длинный/тонкий
D) Other (please describe after [Answer]: tag below)

[B]: 

### Question 3
Как обрабатывать ситуацию, когда объект не распознан (не подходит ни под одну категорию)?

A) Вывести "Неизвестный объект" с указанием обнаруженного цвета и размера
B) Вывести ближайшую категорию с низкой уверенностью (< 50%)
C) Оба варианта: показать ближайшую категорию + пометку "Неизвестный объект"
D) Other (please describe after [Answer]: tag below)

[C]: 

---
