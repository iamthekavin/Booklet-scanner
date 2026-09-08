"""
Interactive demo script for the VEE Scanner Booklet Detector.

Generates synthetic test frames simulating real failure cases and shows
before/after detection results.

Usage:
    python demo.py                          # generate + process synthetic frames
    python demo.py --input path/to/images/  # process real images
    python demo.py --generate-only          # only generate synthetic test frames
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from detector import BookletDetector
from config import DetectorConfig
from models import DetectionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Synthetic Frame Generation ──────────────────────────────────────────────

def generate_wood_texture(width: int, height: int) -> np.ndarray:
    """Generate a realistic wood grain texture using layered noise.

    Uses horizontal streaks with varying brown tones and grain variation.
    """
    base_color = np.array([75, 115, 155], dtype=np.float32)  # BGR warm brown

    # Horizontal grain using stretched noise
    noise = np.random.normal(0, 15, (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (51, 5), 0)  # stretch horizontally

    # Sinusoidal grain lines
    y_coords = np.arange(height).reshape(-1, 1).astype(np.float32)
    grain = np.sin(y_coords / 15.0 + noise / 20.0) * 20.0

    texture = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        channel_noise = np.random.normal(0, 5, (height, width)).astype(np.float32)
        texture[:, :, i] = base_color[i] + grain + channel_noise

    # Add some knots
    for _ in range(3):
        kx = np.random.randint(50, width - 50)
        ky = np.random.randint(50, height - 50)
        kr = np.random.randint(15, 40)
        cv2.circle(texture, (kx, ky), kr, (55, 90, 120), -1)
        texture[max(0,ky-kr):ky+kr, max(0,kx-kr):kx+kr] = cv2.GaussianBlur(
            texture[max(0,ky-kr):ky+kr, max(0,kx-kr):kx+kr], (15, 15), 0
        )

    return np.clip(texture, 0, 255).astype(np.uint8)


def generate_booklet_image(width: int, height: int) -> np.ndarray:
    """Generate a booklet/exam answer sheet with ruled lines and margin.

    Creates a realistic-looking ruled booklet page.
    """
    # Off-white base
    booklet = np.full((height, width, 3), (238, 243, 243), dtype=np.uint8)

    # Red margin line
    margin_x = int(width * 0.15)
    cv2.line(booklet, (margin_x, 0), (margin_x, height), (140, 140, 255), 2)
    cv2.line(booklet, (margin_x + 5, 0), (margin_x + 5, height), (180, 180, 255), 1)

    # Blue/gray ruled horizontal lines
    line_spacing = 30
    start_y = int(height * 0.08)
    for y in range(start_y, height - 20, line_spacing):
        cv2.line(booklet, (15, y), (width - 15, y), (210, 195, 195), 1)

    # Simulated handwriting scribbles
    for _ in range(15):
        pts = []
        cx = np.random.randint(margin_x + 20, width - 60)
        cy = np.random.randint(start_y + 10, height - 60)
        for _ in range(6):
            pts.append([cx + np.random.randint(-15, 15),
                        cy + np.random.randint(-8, 8)])
            cx += np.random.randint(5, 20)
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(booklet, [pts_arr], False, (80, 40, 40), 2)

    return booklet


def _place_booklet_on_background(
    booklet: np.ndarray,
    bg: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Place a booklet image onto a background with rotation.

    Args:
        booklet: Booklet image.
        bg: Background image.
        cx, cy: Center position on background.
        angle_deg: Rotation angle in degrees.

    Returns:
        Composite image.
    """
    bh, bw = booklet.shape[:2]
    fh, fw = bg.shape[:2]

    hw, hh = bw // 2, bh // 2
    corners_local = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
    ], dtype=np.float32)

    # Rotation
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners_rot = (R @ corners_local.T).T

    # Random perspective jitter
    corners_rot += np.random.uniform(-10, 10, (4, 2)).astype(np.float32)

    # Translate to center
    corners_frame = corners_rot + np.array([cx, cy], dtype=np.float32)

    # Perspective warp booklet onto background
    src_pts = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, corners_frame)

    warped = cv2.warpPerspective(booklet, M, (fw, fh))
    mask = cv2.warpPerspective(
        np.ones((bh, bw), dtype=np.uint8) * 255, M, (fw, fh)
    )

    # Composite
    mask_3ch = cv2.merge([mask, mask, mask])
    bg_masked = cv2.bitwise_and(bg, cv2.bitwise_not(mask_3ch))
    return cv2.add(bg_masked, warped)


def generate_test_frame(
    scenario: str, frame_w: int = 1280, frame_h: int = 960
) -> np.ndarray:
    """Generate a complete test frame for a given failure scenario.

    Args:
        scenario: One of 'clean', 'wood_grain', 'shadow', 'low_contrast',
                  'hand_occlusion', 'clutter'.
        frame_w: Frame width.
        frame_h: Frame height.

    Returns:
        BGR image simulating the scenario.
    """
    cx, cy = frame_w // 2, frame_h // 2
    booklet_w = int(frame_w * 0.5)
    booklet_h = int(frame_h * 0.7)
    angle = np.random.uniform(-12, 12)

    # ── Background ──
    if scenario == "wood_grain":
        bg = generate_wood_texture(frame_w, frame_h)
    elif scenario == "low_contrast":
        # Light gray desk — booklet will be hard to see
        bg = np.full((frame_h, frame_w, 3), (210, 210, 210), dtype=np.uint8)
        bg_noise = np.random.normal(0, 6, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)
    else:
        # Standard dark desk
        bg = np.full((frame_h, frame_w, 3), (50, 50, 60), dtype=np.uint8)
        bg_noise = np.random.normal(0, 4, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)

    # ── Booklet ──
    booklet = generate_booklet_image(booklet_w, booklet_h)
    if scenario == "low_contrast":
        # Make booklet closer to background color
        booklet = cv2.addWeighted(booklet, 0.6, np.full_like(booklet, 200), 0.4, 0)

    # Place booklet on background
    frame = _place_booklet_on_background(booklet, bg, cx, cy, angle)

    # ── Scenario-specific effects ──
    if scenario == "shadow":
        # Diagonal gradient shadow across the frame
        shadow = np.ones_like(frame, dtype=np.float32)
        y_grid, x_grid = np.mgrid[0:frame_h, 0:frame_w]
        shadow_mask = ((x_grid + y_grid).astype(np.float32)
                       / (frame_w + frame_h) * 0.6 + 0.4)
        for i in range(3):
            shadow[:, :, i] = shadow_mask
        frame = (frame.astype(np.float32) * shadow).astype(np.uint8)

        # Hard shadow edge near spine
        shadow_x = cx - 10
        frame[:, shadow_x:shadow_x + 20] = (
            frame[:, shadow_x:shadow_x + 20].astype(np.float32) * 0.4
        ).astype(np.uint8)

    elif scenario == "hand_occlusion":
        # Flesh-colored hand polygon
        hand_pts = np.array([
            [cx + 80, cy + 120],
            [cx + 30, cy + 30],
            [cx + 90, cy - 20],
            [cx + 140, cy],
            [cx + 160, cy + 60],
            [cx + 220, cy + 170],
        ], np.int32)
        cv2.fillPoly(frame, [hand_pts], (155, 175, 215))
        # Fingers
        for dx, dy in [(90, -20), (110, -30), (130, -15)]:
            finger_end = (cx + dx, cy + dy - 40)
            finger_base = (cx + dx, cy + dy)
            cv2.line(frame, finger_base, finger_end, (155, 175, 215), 18)
            cv2.circle(frame, finger_end, 9, (155, 175, 215), -1)
        # Pen
        cv2.line(frame, (cx + 30, cy + 30), (cx - 80, cy - 30), (40, 40, 40), 6)

    elif scenario == "clutter":
        # Scattered paper scraps
        for _ in range(4):
            px = np.random.randint(50, frame_w - 200)
            py = np.random.randint(50, frame_h - 200)
            pw = np.random.randint(60, 150)
            ph = np.random.randint(40, 120)
            scrap_color = np.random.randint(200, 255, 3).tolist()
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), scrap_color, -1)
        # Pens / pencils
        for _ in range(3):
            px = np.random.randint(100, frame_w - 100)
            py = np.random.randint(100, frame_h - 100)
            length = np.random.randint(80, 200)
            angle_r = np.random.uniform(0, np.pi)
            dx = int(length * np.cos(angle_r))
            dy = int(length * np.sin(angle_r))
            color = (
                np.random.randint(0, 80),
                np.random.randint(0, 80),
                np.random.randint(0, 80),
            )
            cv2.line(frame, (px, py), (px + dx, py + dy), color, 6)

    return frame


def generate_all_test_frames(output_dir: Path) -> List[Path]:
    """Generate synthetic test frames for all scenarios.

    Args:
        output_dir: Directory to save generated frames.

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        "clean", "wood_grain", "shadow",
        "low_contrast", "hand_occlusion", "clutter",
    ]

    saved_paths = []
    for scenario in scenarios:
        logger.info(f"  Generating: {scenario}")
        frame = generate_test_frame(scenario)
        out_path = output_dir / f"test_{scenario}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

    logger.info(f"  Generated {len(saved_paths)} test frames in {output_dir}")
    return saved_paths


# ─── Visualization ───────────────────────────────────────────────────────────

def draw_detection_overlay(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw detection results (bbox, corners, info) on a copy of the frame.

    Args:
        frame: Original BGR image.
        result: DetectionResult from the detector.

    Returns:
        Annotated image.
    """
    vis = frame.copy()

    if result.corners is None:
        cv2.putText(
            vis, "NO DETECTION", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
        )
        return vis

    # Draw bounding box (green)
    if result.bbox is not None:
        x1, y1 = int(result.bbox.x1), int(result.bbox.y1)
        x2, y2 = int(result.bbox.x2), int(result.bbox.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw corners and edges
    corners = result.corners.points.astype(np.int32)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(vis, p1, p2, (255, 255, 0), 2)  # cyan edges
        cv2.circle(vis, p1, 6, (0, 0, 255), -1)  # red corner dots

    # Info text
    info_lines = [
        f"Method: {result.detection_method.value}",
        f"Confidence: {result.confidence:.3f}",
        f"Latency: {result.latency_ms:.1f}ms",
        f"Review: {'YES' if result.needs_review else 'no'}",
    ]
    flags_str = ", ".join(f.value for f in result.review_flags)
    info_lines.append(f"Flags: {flags_str}")

    if result.hands_detected:
        info_lines.append(f"Hands: {len(result.hands_detected)}")

    y_pos = 35
    for text in info_lines:
        cv2.putText(
            vis, text, (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
        )
        y_pos += 28

    return vis


def make_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    warped: np.ndarray | None,
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Args:
        original: Original frame (labeled "Before").
        annotated: Frame with detection overlay.
        warped: Warped booklet output (may be None).

    Returns:
        Combined comparison image.
    """
    h = original.shape[0]

    # Add "BEFORE" label
    before = original.copy()
    cv2.putText(before, "BEFORE", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Add "DETECTED" label to annotated
    cv2.putText(annotated, "DETECTED", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    panels = [before, annotated]

    if warped is not None:
        # Resize warped to match height
        scale = h / warped.shape[0]
        w_new = int(warped.shape[1] * scale)
        warped_resized = cv2.resize(warped, (w_new, h))
        cv2.putText(warped_resized, "WARPED", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        panels.append(warped_resized)

    return np.hstack(panels)


# ─── Processing ──────────────────────────────────────────────────────────────

def process_image(
    detector: BookletDetector,
    image_path: Path,
    output_dir: Path,
) -> None:
    """Process a single image: detect, visualize, save comparison.

    Args:
        detector: Initialized BookletDetector.
        image_path: Path to input image.
        output_dir: Directory to save results.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"Failed to load: {image_path}")
        return

    # Run detection
    result, warped = detector.detect_and_warp(frame)

    # Visualize
    annotated = draw_detection_overlay(frame, result)
    comparison = make_side_by_side(frame, annotated, warped)

    # Save
    out_path = output_dir / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), comparison)

    # Print summary
    flags_str = ", ".join(f.value for f in result.review_flags)
    print(
        f"  {image_path.name:30s} | "
        f"method={result.detection_method.value:20s} | "
        f"conf={result.confidence:.3f} | "
        f"latency={result.latency_ms:6.1f}ms | "
        f"review={str(result.needs_review):5s} | "
        f"flags=[{flags_str}]"
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VEE Scanner Booklet Detector Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                          # generate + process synthetic frames
  python demo.py --input path/to/images/  # process real images
  python demo.py --generate-only          # only generate synthetic test frames
        """,
    )
    parser.add_argument(
        "--input", type=str,
        help="Path to directory with real images to process",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate synthetic test frames, don't run detection",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "test_images"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # ── Determine images to process ──
    images: List[Path] = []

    if args.input:
        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)
        images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
        if not images:
            logger.error(f"No .jpg/.png images found in {input_dir}")
            sys.exit(1)
    else:
        # Check if synthetic frames already exist
        existing = sorted(test_dir.glob("test_*.jpg"))
        if not existing or args.generate_only:
            logger.info("Generating synthetic test frames...")
            images = generate_all_test_frames(test_dir)
        else:
            images = existing

    if args.generate_only:
        logger.info("Generation complete. Exiting.")
        return

    # ── Initialize detector ──
    print("\n" + "=" * 90)
    print("  VEE Scanner — Booklet Detection Demo")
    print("=" * 90)

    logger.info("Initializing BookletDetector...")
    config = DetectorConfig()
    detector = BookletDetector(config)
    model_info = detector.get_model_info()
    print(f"  Model: {model_info['model_path']}")
    print(f"  Device: {model_info['device']}")
    print(f"  Custom model: {model_info['is_custom_model']}")
    print()

    # ── Process each image ──
    print(f"{'Image':32s} | {'Method':22s} | {'Conf':5s}  | {'Latency':8s} | {'Review':6s} | Flags")
    print("-" * 90)

    for img_path in images:
        process_image(detector, img_path, output_dir)

    print("-" * 90)
    print(f"\n  Results saved to: {output_dir}")
    print(f"  Processed {len(images)} images.\n")


if __name__ == "__main__":
    main()
