# Application Design Plan — Smart Storage

## Plan Steps

- [x] 1. Define core pipeline components and responsibilities
- [x] 2. Define GUI component structure
- [x] 3. Define support components (video processor, CSV exporter, classification engine)
- [x] 4. Generate components.md
- [x] 5. Generate component-methods.md
- [x] 6. Generate services.md
- [x] 7. Generate component-dependency.md
- [x] 8. Generate consolidated application-design.md
- [x] 9. Validate design completeness and consistency

---

## Design Questions

Пожалуйста, ответьте на вопросы ниже, заполнив букву после тега `[Answer]:`.

### Question 1
Какую архитектуру GUI предпочитаете?

A) Tkinter — стандартная библиотека Python, простое окно с панелями для каждого этапа пайплайна
B) OpenCV highgui (cv2.imshow) — минимальный подход, несколько окон для каждого этапа
C) PyQt5/PyQt6 — более продвинутый GUI с вкладками и панелями
D) Other (please describe after [Answer]: tag below)

[B]: 

### Question 2
Как должны быть организованы модули пайплайна?

A) Каждый этап — отдельный класс с единым интерфейсом (паттерн Pipeline/Chain): `process(image) -> result`
B) Каждый этап — отдельная функция в общем модуле `pipeline.py`
C) Один класс `Pipeline` с методами для каждого этапа (`enhance()`, `segment()`, `clean()`, `detect()`, `decide()`)
D) Other (please describe after [Answer]: tag below)

[C]: 

### Question 3
Как организовать конфигурацию правил классификации (цвет + размер → категория)?

A) JSON/YAML файл конфигурации — легко редактировать без изменения кода
B) Python словарь в отдельном модуле `config.py` — просто и прямолинейно
C) Класс `ClassificationRules` с методами для добавления/изменения правил программно
D) Other (please describe after [Answer]: tag below)

[B]: 

---
