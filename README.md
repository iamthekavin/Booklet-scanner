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

# 2. Run the live scanner (webcam or phone camera)
python live_stream.py --source 0 --auto-capture

# Or with an IP camera (e.g. DroidCam/IP Webcam):
python live_stream.py --source http://192.168.1.100:8080/video --auto-capture

# 3. Run benchmarks
python benchmark.py --frames 50

# 4. Run full test suite & PDF verification
python -m pytest -v
python verify_fr42.py
```

## Architecture

```
Camera Stream / Image
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ YOLO11 Detection (Fine-tuned on booklets & hands)         │
│  - Booklet Bounding Box Localization                      │
│  - Multi-background invariant (dark desk, white marble)   │
│  - Hand Detection                                         │
└──────────────────────────┬───────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌───────────────────────────┐    ┌─────────────────────────────────────────┐
│ Confidence-Gated Refiner  │    │ Physical Hand Occlusion Gate            │
│  - Directional Sobel      │    │  - Overlap check against booklet quad   │
│  - Outermost border snap  │    │  - Hands on desk/floor allowed          │
│  - Huber robust line fit  │    │  - Hands occluding booklet pause auto   │
└─────────────┬─────────────┘    └────────────────────┬────────────────────┘
              │                                       │
              └──────────────────┬────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────┐
│ Perspective Warper & Page Splitter                       │
│  - Quad perspective rectification to flat spread        │
│  - Vertical spine crease projection & valley detection   │
│  - Split 2-page spread into left & right portrait pages  │
│  - Contrast enhancement & sharpening                     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Automated Hands-Free Session & PDF Compiler (FR-4.2)     │
│  - Motion & stability tracking                           │
│  - Page-turn detection & perceptual hash de-duplication  │
│  - Direct compilation into single-page portrait PDF      │
└──────────────────────────────────────────────────────────┘
```

## Live Scanner Key Controls

When running `live_stream.py`:
- `SPACE`: Manually trigger capture of the current detected booklet
- `A`: Toggle hands-free auto-capture mode on/off
- `F` / `F11`: Toggle full screen
- `R`: Reset auto-capture state
- `C`: Finish booklet scanning and compile session PDF immediately
- `Q` / `ESC`: Quit and save compiled PDF

## CLI Options

```bash
python live_stream.py [OPTIONS]

Options:
  -s, --source TEXT       Camera source: webcam index (e.g. 0, 1) or stream URL (default: 0)
  -a, --auto-capture      Enable hands-free auto-capture mode from start
  -n, --frames-per-booklet INT
                          Number of spreads per booklet before auto-compiling PDF (default: none)
  -c, --confidence FLOAT  Detection confidence threshold (default: 0.35)
  -d, --device TEXT       Device: auto, cpu, cuda (default: auto)
  --output-dir PATH       Directory for session output (default: output/)
```

## Key Capabilities

1. **Multi-Background Invariance**:
   Fine-tuned on diverse backgrounds including dark wood grain, white marble, cardboard, and low-contrast desk surfaces.

2. **Geometric Hand Occlusion Gating**:
   Distinguishes hands resting on the desk/floor from hands actively occluding the booklet. Capture proceeds smoothly with hands in frame as long as the booklet surface is clear.

3. **Outermost Border Alignment**:
   Sobel gradient peak detection with asymmetric outward search prevents snapping to printed margin lines inside the booklet, ensuring the full page edge is captured.

4. **FR-4.2 Multi-Page PDF Compilation**:
   Each captured 2-page spread is automatically split at the detected spine crease into consecutive portrait pages (e.g. `001.jpg`, `002.jpg`) and compiled into a single PDF document.

## File Structure

```
├── config.py              # Central configuration (thresholds, paths, model settings)
├── detector.py            # BookletDetector class with geometric occlusion gating
├── models.py              # Data classes (DetectionResult, BoundingBox, QuadCorners)
├── corner_refiner.py      # Multi-strategy corner refinement engine (Sobel + Huber)
├── perspective.py         # Perspective warp + spine crease page splitting
├── pdf_compiler.py        # PDF compilation from split pages (FR-4.2 compliant)
├── live_stream.py         # Live camera stream detection, dual-panel HUD, auto/manual capture
├── auto_capture.py        # Automated hands-free capture controller (stability + anti-duplicate)
├── demo.py                # Visual demo with before/after comparison
├── benchmark.py           # Latency benchmarking
├── verify_fr42.py         # FR-4.2 PDF compilation verification script
├── test_detector.py       # Detection unit tests (38 tests)
├── test_pdf_compiler.py   # PDF compilation unit tests (8 tests)
├── test_auto_capture.py   # Auto-capture unit tests (13 tests)
├── requirements.txt       # Dependencies
│
├── training/
│   ├── dataset.yaml       # YOLO dataset configuration
│   ├── train.py           # Custom model training script
│   ├── augment_backgrounds.py # Multi-background augmentation script
│   ├── generate_synthetic.py  # Synthetic training data generator
│   ├── dataset_guide.md   # Annotation guide for labelers
│   └── assets/            # Training assets (backgrounds/booklets)
│
├── models/                # Trained model weights (booklet_detector.pt, ONNX)
├── output/                # Session output images and compiled PDFs (git-ignored)
└── test_images/           # Test frames for validation
```

## Testing & Verification

```bash
# Run all 59 automated unit tests
python -m pytest

# Run end-to-end FR-4.2 verification
python verify_fr42.py
```

## License

Private — VEE Scanner project.

