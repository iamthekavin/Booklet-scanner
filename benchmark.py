"""
VEE Scanner Benchmark Script
Measures detection pipeline latency across multiple frames.

Usage:
    python benchmark.py --frames 50
    python benchmark.py --frames 100 --model yolo11s.pt
    python benchmark.py --frames 50 --device cuda
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
from typing import Dict, List

import cv2
import numpy as np

from config import DetectorConfig
from detector import BookletDetector


def generate_synthetic_frame(width: int = 1280, height: int = 960) -> np.ndarray:
    """Generate a synthetic test frame with a booklet on a textured background.

    Args:
        width: Frame width.
        height: Frame height.

    Returns:
        BGR image.
    """
    # Dark desk background with some noise
    frame = np.full((height, width, 3), (60, 55, 50), dtype=np.uint8)
    noise = np.random.normal(0, 8, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw a white booklet rectangle
    cx, cy = width // 2, height // 2
    bw, bh = 600, 850
    pts = np.array([
        [cx - bw // 2, cy - bh // 2],
        [cx + bw // 2, cy - bh // 2],
        [cx + bw // 2, cy + bh // 2],
        [cx - bw // 2, cy + bh // 2],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (235, 235, 240))

    # Add some ruled lines on the booklet
    for y in range(cy - bh // 2 + 40, cy + bh // 2, 30):
        cv2.line(
            frame,
            (cx - bw // 2 + 30, y),
            (cx + bw // 2 - 30, y),
            (200, 200, 210), 1,
        )

    return frame


def format_ms(val_seconds: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{val_seconds * 1000:.0f}ms"


def calculate_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate summary statistics for a list of latency values (in seconds)."""
    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_table(
    model_name: str,
    device: str,
    num_frames: int,
    stats: Dict[str, Dict[str, float]],
) -> None:
    """Print a formatted benchmark results table."""
    print("\n" + "=" * 60)
    print("  VEE Scanner Pipeline Benchmark")
    print("=" * 60)
    print(f"  Model:  {model_name}")
    print(f"  Device: {device}")
    print(f"  Frames: {num_frames}")
    print()

    print("┌──────────────────┬─────────┬─────────┬─────────┬─────────┐")
    print("│ Metric           │ Mean    │ Median  │ P95     │ P99     │")
    print("├──────────────────┼─────────┼─────────┼─────────┼─────────┤")

    for metric, values in stats.items():
        name = metric.ljust(16)
        mean_s = format_ms(values["mean"]).ljust(7)
        med_s = format_ms(values["median"]).ljust(7)
        p95_s = format_ms(values["p95"]).ljust(7)
        p99_s = format_ms(values["p99"]).ljust(7)
        print(f"│ {name} │ {mean_s} │ {med_s} │ {p95_s} │ {p99_s} │")

    print("└──────────────────┴─────────┴─────────┴─────────┴─────────┘")
    print()

    # Pass/fail against target
    target_ms = 150
    median_ms = stats["Total Pipeline"]["median"] * 1000
    status = "✓ PASS" if median_ms < target_ms else (
        "~ WARN" if median_ms < 800 else "✗ FAIL"
    )
    print(f"  Target: <{target_ms}ms ideal, <800ms max")
    print(f"  Result: {status} (median {median_ms:.0f}ms)")
    print()


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the benchmark."""
    print(f"\n🔧 Initializing detector with model={args.model}, device={args.device}")
    config = DetectorConfig(model_path=args.model, device=args.device)
    detector = BookletDetector(config)

    model_info = detector.get_model_info()
    print(f"   Model loaded: {model_info}")

    # Generate test frames
    print(f"\n📐 Generating {args.frames} synthetic test frames...")
    frames = [generate_synthetic_frame() for _ in range(args.frames)]

    # Warmup (3 frames)
    print("🔥 Warming up (3 frames)...")
    for i in range(min(3, args.frames)):
        detector.detect(frames[i])

    # Benchmark
    print(f"⏱  Running benchmark ({args.frames} frames)...")
    total_times = []
    detection_counts = {"yolo_direct": 0, "yolo_cv_refined": 0, "cv_fallback": 0, "none": 0}
    review_count = 0

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        result = detector.detect(frame)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_times.append(elapsed)

        if result is not None:
            detection_counts[result.detection_method.value] = (
                detection_counts.get(result.detection_method.value, 0) + 1
            )
            if result.needs_review:
                review_count += 1

        if (i + 1) % 10 == 0:
            print(f"   ... {i + 1}/{args.frames} frames processed")

    # Calculate stats
    stats = {
        "Total Pipeline": calculate_stats(total_times),
    }

    # Print results
    print_table(args.model, args.device, args.frames, stats)

    # Detection method breakdown
    print("  Detection Methods:")
    for method, count in detection_counts.items():
        pct = count / args.frames * 100
        print(f"    {method}: {count}/{args.frames} ({pct:.0f}%)")
    print(f"    Flagged for review: {review_count}/{args.frames}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="VEE Scanner Pipeline Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frames", type=int, default=20,
        help="Number of frames to benchmark (default: 20)",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt",
        help="Path to YOLO model weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu, cuda, cuda:0 (default: cpu)",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
