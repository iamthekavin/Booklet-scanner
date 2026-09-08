"""
Synthetic Data Generator for VEE Scanner.

Composites booklets onto desk backgrounds to bootstrap training data.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_procedural_background(size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
    """Generate a procedural wood-grain-like texture."""
    # Create base noise
    noise = np.random.randint(0, 255, (size[1] // 4, size[0] // 4), dtype=np.uint8)
    noise = cv2.resize(noise, size, interpolation=cv2.INTER_LINEAR)
    
    # Apply motion blur for wood grain effect
    kernel_size = 31
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    grain = cv2.filter2D(noise, -1, kernel)
    
    # Add color (brownish)
    bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    bg[:, :, 0] = grain * 0.3 + 50  # B
    bg[:, :, 1] = grain * 0.5 + 80  # G
    bg[:, :, 2] = grain * 0.6 + 120 # R
    
    # Add some random brightness variation
    shadow = np.random.uniform(0.7, 1.2)
    bg = np.clip(bg * shadow, 0, 255).astype(np.uint8)
    return bg


def create_procedural_booklet(size: Tuple[int, int] = (600, 800)) -> np.ndarray:
    """Generate a synthetic white/off-white booklet with ruled lines."""
    # Off-white paper color
    color = (np.random.randint(230, 255), np.random.randint(230, 255), np.random.randint(230, 255))
    booklet = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    
    # Draw ruled lines (blue)
    line_spacing = 30
    line_color = (150, 100, 50)  # Blue-ish BGR
    for y in range(80, size[1] - 40, line_spacing):
        cv2.line(booklet, (40, y), (size[0] - 40, y), line_color, 2)
        
    # Draw margin line (red)
    margin_color = (50, 50, 200) # Red-ish BGR
    cv2.line(booklet, (80, 0), (80, size[1]), margin_color, 2)
    
    return booklet


def get_random_perspective_transform(w: int, h: int, max_distort: float = 0.1) -> np.ndarray:
    """Get a random perspective transform matrix."""
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Perturb points
    dx = w * max_distort
    dy = h * max_distort
    pts2 = np.float32([
        [random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
        [random.uniform(-dx, dx), h + random.uniform(-dy, dy)]
    ])
    
    return cv2.getPerspectiveTransform(pts1, pts2)


def generate_sample(bg_path: Path | None, booklet_path: Path | None, out_size: Tuple[int, int] = (1280, 720)) -> Tuple[np.ndarray, List[float]]:
    """Generate one synthetic sample and its YOLO bounding box (cx, cy, w, h)."""
    
    # 1. Get background
    if bg_path and bg_path.exists():
        bg = cv2.imread(str(bg_path))
        if bg is None:
            bg = create_procedural_background(out_size)
        else:
            bg = cv2.resize(bg, out_size)
    else:
        bg = create_procedural_background(out_size)
        
    # 2. Get booklet
    if booklet_path and booklet_path.exists():
        booklet = cv2.imread(str(booklet_path))
        if booklet is None:
            booklet = create_procedural_booklet()
    else:
        booklet = create_procedural_booklet()
        
    bh, bw = booklet.shape[:2]
    
    # 3. Random scale (booklet occupies 30-80% of frame height)
    target_h = int(out_size[1] * random.uniform(0.3, 0.8))
    scale = target_h / bh
    target_w = int(bw * scale)
    booklet = cv2.resize(booklet, (target_w, target_h))
    bh, bw = target_h, target_w
    
    # 4. Perspective distortion
    M_persp = get_random_perspective_transform(bw, bh)
    booklet = cv2.warpPerspective(booklet, M_persp, (bw, bh), borderMode=cv2.BORDER_TRANSPARENT)
    
    # 5. Rotation (-20 to 20 degrees)
    angle = random.uniform(-20, 20)
    center = (bw // 2, bh // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box after rotation to avoid cropping
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    new_w = int((bh * sin) + (bw * cos))
    new_h = int((bh * cos) + (bw * sin))
    M_rot[0, 2] += (new_w / 2) - center[0]
    M_rot[1, 2] += (new_h / 2) - center[1]
    
    booklet_rot = cv2.warpAffine(booklet, M_rot, (new_w, new_h), borderValue=(0, 0, 0))
    
    # Create mask for blending
    mask = cv2.cvtColor(booklet_rot, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.merge([mask, mask, mask])
    
    # 6. Composite onto background at random position
    max_x = max(0, out_size[0] - new_w)
    max_y = max(0, out_size[1] - new_h)
    
    x_offset = random.randint(0, max_x)
    y_offset = random.randint(0, max_y)
    
    # Bounding box limits
    x1, y1 = x_offset, y_offset
    x2, y2 = x_offset + new_w, y_offset + new_h
    
    # Add shadow
    shadow_offset = random.randint(5, 15)
    shadow_mask = np.zeros_like(bg, dtype=np.uint8)
    shadow_y1, shadow_y2 = min(out_size[1], y1 + shadow_offset), min(out_size[1], y2 + shadow_offset)
    shadow_x1, shadow_x2 = min(out_size[0], x1 + shadow_offset), min(out_size[0], x2 + shadow_offset)
    
    if shadow_y2 > shadow_y1 and shadow_x2 > shadow_x1:
        s_h, s_w = shadow_y2 - shadow_y1, shadow_x2 - shadow_x1
        shadow_mask[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = mask[:s_h, :s_w]
    
    # Darken background where shadow is
    bg = np.where(shadow_mask > 0, (bg * 0.7).astype(np.uint8), bg)
    
    # Blend booklet
    roi = bg[y1:y2, x1:x2]
    bg[y1:y2, x1:x2] = np.where(mask > 0, booklet_rot, roi)
    
    # Normalize YOLO bbox (cx, cy, w, h)
    cx = (x1 + x2) / 2.0 / out_size[0]
    cy = (y1 + y2) / 2.0 / out_size[1]
    nw = new_w / out_size[0]
    nh = new_h / out_size[1]
    
    return bg, [cx, cy, nw, nh]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic booklet training data.")
    parser.add_argument('--backgrounds', type=str, help='Directory of background images')
    parser.add_argument('--booklets', type=str, help='Directory of booklet images')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--count', type=int, default=500, help='Number of images to generate')
    
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    images_train = out_dir / 'images' / 'train'
    labels_train = out_dir / 'labels' / 'train'
    
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    
    bg_files = list(Path(args.backgrounds).rglob('*.jpg')) if args.backgrounds else []
    booklet_files = list(Path(args.booklets).rglob('*.jpg')) if args.booklets else []
    
    logging.info(f"Starting generation of {args.count} synthetic images...")
    
    for i in range(args.count):
        bg_path = random.choice(bg_files) if bg_files else None
        bk_path = random.choice(booklet_files) if booklet_files else None
        
        img, bbox = generate_sample(bg_path, bk_path)
        
        # Save image
        img_name = f"synth_{i:05d}.jpg"
        cv2.imwrite(str(images_train / img_name), img)
        
        # Save label (class 0: booklet)
        label_name = f"synth_{i:05d}.txt"
        with open(labels_train / label_name, 'w') as f:
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
        if (i + 1) % 50 == 0:
            logging.info(f"Generated {i + 1}/{args.count} samples.")

    logging.info(f"Finished generating {args.count} samples in {out_dir}")


if __name__ == '__main__':
    main()
