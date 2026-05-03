# Failure Cases — Smart Storage

## FC-01: Coiled Cable vs Wired Earphones (White/Silver)

**Status**: Known limitation — not fixable with rule-based approach

**Description**:
The system cannot reliably distinguish a coiled white/silver USB-C cable from white wired earphones (e.g., iPhone EarPods) when both are photographed in a coiled/bundled state.

**Root Cause**:
Both objects share identical visual features when coiled:
- Color: white or silver
- Shape: ring/loop with irregular contour
- Circularity: ~0.30–0.45 (ring shape)
- Solidity: ~0.80–0.90 (cleaned mask fills the center hole)
- Size: medium

The rule-based classifier uses color + shape + size features. Since these features are identical for both objects in coiled state, the system cannot differentiate them.

**Observed Behavior**:
- Input: coiled silver USB-C cable on beige background
- Detected features: circ=0.351, solid=0.832, extent=0.577, color=silver, size=medium
- System output: "Headphones" (incorrect) or "USB-C Cable" (correct by chance)

**Workaround**:
- Present cables in a **straight/uncoiled** state → system correctly identifies as `long_thin` → USB-C Cable or Power Cable
- Straight cables have aspect_ratio > 4.0 which uniquely identifies them as cables

**Potential Fix (future work)**:
- Use deep learning (CNN) for shape classification instead of geometric features
- Train on labeled dataset of coiled cables vs earphones
- Add texture analysis (cable braid texture vs smooth earphone wire)

---

## FC-02: Non-White/Non-Black Background

**Status**: Known limitation

**Description**:
The segmentation pipeline works best on solid white or black backgrounds. On beige, brown, or textured backgrounds, the Otsu threshold may capture the background instead of the object.

**Observed Behavior**:
- Input: cable on beige/brown desk surface
- Segmentation mask captures background (area_ratio > 0.60)
- System applies mask inversion as fallback, but result may be noisy

**Workaround**:
- Use a white sheet of paper as background for best results
- Black background also works well for light-colored objects

---

## FC-03: Dark Object on Dark Background

**Status**: Known limitation

**Description**:
Dark objects (black mouse, dark keyboard) on dark backgrounds cannot be segmented — insufficient contrast for threshold-based methods.

**Workaround**:
- Always use contrasting background (dark object → white background, light object → dark background)

---

## FC-04: Multiple Objects in Frame

**Status**: By design — single object mode

**Description**:
The system is designed for one object at a time. When multiple objects are present, the pipeline selects the largest contour and ignores others.

**Workaround**:
- Place one object at a time in the frame
- Future enhancement: multi-object detection with bounding boxes per object
