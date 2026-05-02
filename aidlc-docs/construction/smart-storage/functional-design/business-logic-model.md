# Business Logic Model — Smart Storage

## Pipeline Stage 1: Enhancement

### Algorithm: Contrast + Denoising
```
Input: BGR image (raw from camera)
Steps:
  1. Convert to LAB color space
  2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
     - clipLimit: 2.0
     - tileGridSize: (8, 8)
  3. Convert back to BGR
  4. Apply Gaussian blur for denoising
     - kernel: (5, 5)
     - sigmaX: 0
Output: Enhanced BGR image
```

### Purpose
- Normalize lighting conditions across different environments
- Improve contrast between object and background
- Reduce camera noise for cleaner segmentation

---

## Pipeline Stage 2: Segmentation

### Algorithm: Adaptive Threshold
```
Input: Enhanced BGR image
Steps:
  1. Convert to grayscale
  2. Apply GaussianBlur (kernel 11x11) to reduce noise
  3. Apply adaptiveThreshold:
     - method: ADAPTIVE_THRESH_GAUSSIAN_C
     - type: THRESH_BINARY_INV
     - blockSize: 11
     - C: 2
  4. Result: binary mask (object = 255, background = 0)
Output: Binary segmentation mask
```

### Why Adaptive Threshold
- Better than Otsu for uneven lighting on desk surfaces
- Fast enough for real-time video processing
- Works well with solid-color backgrounds (white/black)

---

## Pipeline Stage 3: Mask Cleaning

### Algorithm: Morphological Operations
```
Input: Binary segmentation mask
Steps:
  1. Morphological CLOSE (fill small holes inside object)
     - kernel: elliptical, size (7, 7)
     - iterations: 2
  2. Morphological OPEN (remove small noise outside object)
     - kernel: elliptical, size (5, 5)
     - iterations: 2
  3. Find contours
  4. Keep only the largest contour (main object)
  5. Create clean mask from largest contour only
Output: Cleaned binary mask (single object)
```

### Purpose
- Remove segmentation artifacts and noise
- Fill holes inside the object
- Isolate the single largest object (ignore small debris)

---

## Pipeline Stage 4: Detection

### Algorithm: Contour Analysis + Color Detection
```
Input: Enhanced image + Cleaned mask
Steps:
  1. Find contours on cleaned mask
  2. Get bounding rectangle of largest contour
     - bbox = cv2.boundingRect(contour) -> (x, y, w, h)
  3. Calculate object properties:
     - area_pixels = cv2.contourArea(contour)
     - aspect_ratio = max(w, h) / min(w, h)
     - perimeter = cv2.arcLength(contour)
  4. Determine size category (see Business Rules)
  5. Run color detection (both methods):
     - HSV analysis on masked region
     - K-means clustering on masked region
  6. Assemble DetectionResult
Output: DetectionResult (bbox, colors, size, area, aspect_ratio)
```

---

## Pipeline Stage 5: Decision

### Algorithm: Rule-Based Classification
```
Input: DetectionResult
Steps:
  1. Get primary color (highest confidence from either method)
  2. Get size category
  3. Look up classification rules table (color + size -> category)
  4. If exact match found:
     - Return category with high confidence
  5. If no exact match:
     - Find closest match (partial color or size match)
     - Return closest category with low confidence + "Unknown" flag
  6. Calculate final confidence score
Output: Decision (category, confidence, color, size, method)
```

---

## Color Detection: HSV Method

### Algorithm
```
Input: BGR image region (masked)
Steps:
  1. Convert masked region to HSV
  2. Calculate histogram of H channel (within mask)
  3. Find dominant hue peak
  4. Map hue to color name using HSV ranges:
     - Check S and V channels for white/black/gray detection
     - Low S + High V = White
     - Low V = Black
     - Low S + Medium V = Gray/Silver
     - Otherwise: map H to color name
  5. Calculate confidence = (dominant_pixels / total_pixels)
Output: ColorResult (name, confidence, method="hsv")
```

### HSV Color Ranges
| Color | H Range | S Range | V Range |
|-------|---------|---------|---------|
| White | any | 0-50 | 200-255 |
| Black | any | 0-255 | 0-50 |
| Gray/Silver | any | 0-50 | 50-200 |
| Red | 0-10, 170-180 | 50-255 | 50-255 |
| Blue | 100-130 | 50-255 | 50-255 |
| Green | 35-85 | 50-255 | 50-255 |

---

## Color Detection: K-Means Method

### Algorithm
```
Input: BGR image region (masked)
Steps:
  1. Extract pixels within mask (flatten to Nx3 array)
  2. Run K-means clustering:
     - n_clusters: 3
     - max_iter: 100
     - n_init: 10
  3. Find largest cluster (most pixels)
  4. Get cluster center (BGR values)
  5. Convert center to HSV
  6. Map to color name using same HSV ranges
  7. Confidence = largest_cluster_size / total_pixels
Output: ColorResult (name, confidence, method="kmeans")
```

---

## Method Comparison Logic

```
Input: ColorResult (hsv), ColorResult (kmeans)
Logic:
  IF both methods agree on color name:
    final_color = agreed color
    final_confidence = max(hsv.confidence, kmeans.confidence)
    method_used = "combined"
  ELSE:
    final_color = color from method with higher confidence
    final_confidence = higher_confidence * 0.8 (penalty for disagreement)
    method_used = method with higher confidence
Output: Primary color for classification
```
