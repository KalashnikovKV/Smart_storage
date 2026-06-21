# Smart Storage: Rule-based vs YOLO-Seg

Comparison report for the IT peripheral recognition pipeline. Both methods share the same color detection (HSV + K-means) after segmentation; they differ in **how the object mask and category are obtained**.

## Executive summary

| Evaluation set | Samples | Rule-based accuracy | YOLO-Seg accuracy |
|---|---:|---:|---:|
| Validation (Roboflow val split) | 52 | **11.5%** | **82.7%** |
| Test photos (folder labels, max 8/class) | 40 | **12.5%** | **85.0%** |

**YOLO model:** `runs/segment/runs/segment/smart_storage_final/weights/best.pt`  
**Training:** YOLO11n-seg, 50 epochs, 260 images (208 train / 52 val), imgsz 416, batch 8, GTX 1660 Ti.  
**Final val metrics (epoch 50):** mask mAP50 = **0.740**, mask mAP50-95 = **0.625**.

### Why rule-based fails on real photos

1. **FC-03** — light object on light background: threshold segmentation returns empty or background-heavy masks.
2. **Shape heuristics (R01–R10)** — oval vs rectangular vs long_thin rules confuse similar silhouettes (cable vs headphones).
3. **Single global threshold** — cannot separate instance from clutter when contrast is low.

### YOLO-Seg advantages

- Instance segmentation trained on **260 manually annotated polygons** (Roboflow).
- Class label comes directly from the detector; color/size still computed inside the mask.
- Robust on charger, mouse, cable, flash drive; weaker on **headphones** and **cable** (few training samples).

---

## Pipeline stages (reference)

Each dashboard panel:

1. Original → 2. Enhanced (CLAHE) → 3. Segmentation mask → 4. Cleaned mask → 5. Detection → 6. Decision

For **Rule-based**, panels 3–4 use Otsu/threshold segmentation + morphology.  
For **YOLO-Seg**, panels 3–4 show the **YOLO instance mask** (threshold path is bypassed for classification).

---

## Case studies

### Charger Adapter on light background (FC-03)

- **Ground truth:** Charger Adapter
- **Note:** Rule-based mis-segments the white adapter; YOLO-Seg finds the correct instance mask.
- **Rule-based:** **Headphones** — silver, medium, confidence 66%, method `combined`
- **YOLO-Seg:** **Charger Adapter** — silver, medium, confidence 100%, method `yolo`

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/01_charger/rule_dashboard.jpg) | ![yolo dashboard](assets/01_charger/yolo_dashboard.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/01_charger/rule_pipeline_strip.jpg) | ![yolo strip](assets/01_charger/yolo_pipeline_strip.jpg) |

---

### USB-C Cable (coiled shape)

- **Ground truth:** USB-C Cable
- **Note:** Long thin object confuses rule heuristics; YOLO classifies by learned shape.
- **Rule-based:** **Charger Adapter** — silver, small, confidence 84%, method `combined`
- **YOLO-Seg:** **USB-C Cable** — silver, large, confidence 74%, method `yolo`

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/02_cable/rule_dashboard.jpg) | ![yolo dashboard](assets/02_cable/yolo_dashboard.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/02_cable/rule_pipeline_strip.jpg) | ![yolo strip](assets/02_cable/yolo_pipeline_strip.jpg) |

---

### Wireless mouse

- **Ground truth:** Mouse
- **Note:** Threshold mask often empty on dark mouse; YOLO detects oval silhouette reliably.
- **Rule-based:** **Unknown (closest: Cable-like or headphones-like object)** — white, large, confidence 27%, method `combined`
- **YOLO-Seg:** **Mouse** — white, large, confidence 74%, method `yolo`

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/03_mouse/rule_dashboard.jpg) | ![yolo dashboard](assets/03_mouse/yolo_dashboard.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/03_mouse/rule_pipeline_strip.jpg) | ![yolo strip](assets/03_mouse/yolo_pipeline_strip.jpg) |

---

### Flash drive

- **Ground truth:** Flash Drive
- **Note:** Small object with low contrast — rule-based returns Unknown; YOLO succeeds.
- **Rule-based:** **Mouse** — black, medium, confidence 73%, method `combined`
- **YOLO-Seg:** **Flash Drive** — black, medium, confidence 63%, method `yolo`

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/04_flash_drive/rule_dashboard.jpg) | ![yolo dashboard](assets/04_flash_drive/yolo_dashboard.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/04_flash_drive/rule_pipeline_strip.jpg) | ![yolo strip](assets/04_flash_drive/yolo_pipeline_strip.jpg) |

---

### Headphones

- **Ground truth:** Headphones
- **Note:** Curved shape similar to cable/charger; rules pick wrong category, YOLO correct.
- **Rule-based:** **USB-C Cable** — black, medium, confidence 30%, method `combined`
- **YOLO-Seg:** **Headphones** — black, medium, confidence 94%, method `yolo`

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/05_headphones/rule_dashboard.jpg) | ![yolo dashboard](assets/05_headphones/yolo_dashboard.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/05_headphones/rule_pipeline_strip.jpg) | ![yolo strip](assets/05_headphones/yolo_pipeline_strip.jpg) |

---

### Charger — both methods agree

- **Ground truth:** Charger Adapter
- **Note:** High-contrast scene where threshold segmentation still works for rules.
- **Rule-based:** **Charger Adapter** — silver, small, confidence 50%, method `combined`
- **YOLO-Seg:** **Charger Adapter** — silver, medium, confidence 98%, method `yolo`

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/06_both_ok/rule_dashboard.jpg) | ![yolo dashboard](assets/06_both_ok/yolo_dashboard.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/06_both_ok/rule_pipeline_strip.jpg) | ![yolo strip](assets/06_both_ok/yolo_pipeline_strip.jpg) |

---

### Headphones — both methods struggle

- **Ground truth:** Headphones
- **Note:** Failure case (FC-01): visually ambiguous pose; YOLO may miss detection entirely.
- **Rule-based:** **Charger Adapter** — silver, small, confidence 84%, method `combined`
- **YOLO-Seg:** No detection

#### Full pipeline dashboard

| Rule-based | YOLO-Seg |
|---|---|
| ![rule dashboard](assets/07_both_fail/rule_dashboard.jpg) | ![yolo — no detection](assets/07_both_fail/yolo_no_detection.jpg) |

#### Masks and detection

| Rule-based (mask → overlay → detection) | YOLO-Seg (instance mask → overlay → detection) |
|---|---|
| ![rule strip](assets/07_both_fail/rule_pipeline_strip.jpg) | ![yolo — no detection](assets/07_both_fail/yolo_no_detection.jpg) |

---

## Per-class accuracy (validation set)

| Class | Rule-based | YOLO-Seg |
|---|---:|---:|
| Charger Adapter | 0/10 | 10/10 |
| USB-C Cable | 2/9 | 9/9 |
| Mouse | 1/10 | 7/10 |
| Flash Drive | 0/10 | 9/10 |
| Headphones | 3/13 | 8/13 |

## Reproduce

Metrics and CSV exports are already in `output/comparison_val_summary.txt`, `output/comparison_test_images_summary.txt`, and the matching `.csv` files.

Run the pipeline on a single photo (rule-based vs YOLO):

```bash
./run.sh --mode image --source training/test_images/Charger_Adapter/Image_43.jpg
./run.sh --mode image --source training/test_images/Charger_Adapter/Image_43.jpg \
  --model runs/segment/runs/segment/smart_storage_final/weights/best.pt --device cuda
```

## Data sources

- Dataset: 260 images exported from Roboflow (Instance Segmentation, 6 classes + colored_object).
- Detailed CSV: `output/comparison_val.csv`, `output/comparison_test_images.csv`.
