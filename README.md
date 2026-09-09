# VEE Scanner — Booklet Boundary Detection Module

**YOLO11-based booklet boundary detection** for the VEE Scanner privacy-first exam scanning system.

Replaces unreliable OpenCV contour-based edge detection with a robust two-stage pipeline:
1. **YOLO11 object detection** → coarse booklet localization (immune to desk texture, shadows, clutter)
2. **Classical CV corner refinement** → precise 4-corner quad estimation within the detected ROI

## Why This Exists

The old contour-based approach failed because:
- Desk wood grain created false contour edges
- Shadows near the spine broke the booklet contour into pieces
- Low contrast between booklet and desk surface caused missed edges
- Clutter (pens, papers) on the desk merged into the detected boundary

The YOLO11 approach solves this by first *localizing* the booklet with a neural network (which is robust to texture/lighting), then running classical CV *only inside the detected ROI* where contour detection is reliable.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo (generates synthetic test frames on first run)
python demo.py

# 3. Run benchmarks
python benchmark.py --frames 50

# 4. Run tests
python -m pytest test_detector.py -v
```

## Architecture

```
Frame → YOLO11 Inference → Confidence Gate → Corner Refinement → Perspective Warp
         │                    │                  │                    │
         ├─ booklet bbox     ├─ high: YOLO+CV   ├─ Hough lines      ├─ flat booklet
         ├─ hand detection   ├─ medium: CV+flag  ├─ contour approx   ├─ page split
         └─ clutter detect   └─ low: flag review └─ corner detect    └─ enhance
```

### Confidence Gate Logic

| Confidence | Action | Needs Review? |
|---|---|---|
| ≥ 0.6 (high) | Trust YOLO bbox + CV corner refinement | No |
| 0.3–0.6 (medium) | YOLO bbox + aggressive CV refinement | Depends on CV quality |
| < 0.3 (low) | Best-effort detection, flag for review | **Yes** |
| No detection | Full-frame CV fallback, flag for review | **Yes** |

## File Structure

```
├── config.py              # Central configuration (thresholds, paths, model settings)
├── detector.py            # Main BookletDetector class
├── models.py              # Data classes (DetectionResult, BoundingBox, QuadCorners)
├── corner_refiner.py      # Multi-strategy corner refinement engine
├── perspective.py         # Perspective warp + page splitting
├── pdf_compiler.py        # PDF compilation from split pages (FR-4.2)
├── live_stream.py         # Live webcam/phone stream detection & capture (auto & manual)
├── auto_capture.py        # Automated hands-free capture controller (stability + anti-duplicate)
├── demo.py                # Visual demo with before/after comparison
├── benchmark.py           # Latency benchmarking
├── verify_fr42.py         # FR-4.2 PDF compilation verification script
├── test_detector.py       # Detection unit tests
├── test_pdf_compiler.py   # PDF compilation unit tests
├── test_auto_capture.py   # Auto-capture unit tests
├── requirements.txt       # Dependencies
│
├── training/
│   ├── dataset.yaml       # YOLO dataset configuration
│   ├── train.py           # Custom model training script
│   ├── generate_synthetic.py  # Synthetic training data generator
│   └── dataset_guide.md   # Annotation guide for labelers
│
├── models/                # Trained model weights (git-ignored)
├── output/                # Demo output images
└── test_images/           # Test frames (generated or real)
```

## Two-Phase Approach

### Phase 1 (Current — Works Now)
Uses COCO-pretrained YOLO11 (`yolo11n.pt`) which knows the `book` class (COCO class 73). Combined with CV corner refinement inside the detected ROI, this handles most real-world scenarios.

### Phase 2 (After Labeling Data)
Train a custom YOLO11 model on your specific booklet images for maximum accuracy:

```bash
# 1. Generate synthetic training data to bootstrap
python training/generate_synthetic.py --count 500

# 2. (Recommended) Label 300-500 real images — see training/dataset_guide.md

# 3. Train custom model
python training/train.py train --model yolo11s.pt --epochs 100

# 4. Use the custom model
# Set custom_model_path in config.py to your trained weights
```

## Performance

| Configuration | Device | Expected Latency |
|---|---|---|
| YOLO11n (nano) | CPU | ~80–150ms |
| YOLO11s (small) | CPU | ~150–300ms |
| YOLO11n (nano) | GPU (CUDA) | ~10–25ms |
| YOLO11n ONNX | CPU | ~50–100ms |
| Full pipeline (detect + refine) | CPU | ~100–200ms |

All configurations meet the 800ms latency budget. The nano model on CPU typically runs under 150ms.

## Usage Example

```python
from detector import BookletDetector
from config import DetectorConfig

# Initialize with defaults
detector = BookletDetector()

# Or customize
config = DetectorConfig(
    confidence_threshold=0.3,
    device='cuda',  # or 'cpu', 'auto'
)
detector = BookletDetector(config)

# Detect booklet in a frame
import cv2
frame = cv2.imread("exam_booklet.jpg")
result = detector.detect(frame)

print(f"Method: {result.detection_method.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Needs review: {result.needs_review}")
print(f"Latency: {result.latency_ms:.1f}ms")

if result.corners is not None:
    # Warp to flat image
    warped = detector.warp(frame, result)
    cv2.imwrite("warped_booklet.jpg", warped)
```

## License

Private — VEE Scanner project.
