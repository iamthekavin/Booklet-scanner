
import os, random, cv2, numpy as np
from pathlib import Path
from typing import List, Tuple

def create_marble_background(size=(640, 640)):
    w, h = size
    base_val = random.randint(160, 205)
    bg = np.full((h, w, 3), base_val, dtype=np.uint8)
    noise = np.random.normal(0, 12, (h // 8, w // 8)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    bg = np.clip(bg.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
    num_veins = random.randint(3, 7)
    for _ in range(num_veins):
        pts = []
        x = random.randint(0, w)
        y = random.randint(0, h)
        for _ in range(random.randint(4, 8)):
            pts.append([x, y])
            x += random.randint(-80, 80)
            y += random.randint(40, 120)
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        vein_color = (base_val - random.randint(20, 45),) * 3
        cv2.polylines(bg, [pts], False, vein_color, random.randint(1, 3), cv2.LINE_AA)
    return cv2.GaussianBlur(bg, (5, 5), 0)

def create_white_desk_background(size=(640, 640)):
    w, h = size
    base_val = random.randint(215, 245)
    bg = np.full((h, w, 3), base_val, dtype=np.uint8)
    grad = np.linspace(0.95, 1.05, h)[:, None, None]
    return np.clip(bg.astype(np.float32) * grad, 0, 255).astype(np.uint8)

def create_dark_wood_background(size=(640, 640)):
    w, h = size
    noise = np.random.randint(0, 255, (h // 4, w // 4), dtype=np.uint8)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    kernel = np.zeros((31, 31))
    kernel[15, :] = 1.0 / 31.0
    grain = cv2.filter2D(noise, -1, kernel)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    tone = random.uniform(0.5, 1.2)
    bg[:, :, 0] = np.clip(grain * 0.25 * tone + 35, 0, 255).astype(np.uint8)
    bg[:, :, 1] = np.clip(grain * 0.45 * tone + 55, 0, 255).astype(np.uint8)
    bg[:, :, 2] = np.clip(grain * 0.55 * tone + 80, 0, 255).astype(np.uint8)
    return bg

def create_colored_mat_background(size=(640, 640)):
    w, h = size
    palette = [(160, 110, 40), (50, 130, 60), (40, 50, 160), (50, 50, 50)]
    color = random.choice(palette)
    bg = np.full((h, w, 3), color, dtype=np.uint8)
    grid_color = tuple(min(255, c + 25) for c in color)
    for x in range(0, w, 40): cv2.line(bg, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, 40): cv2.line(bg, (0, y), (w, y), grid_color, 1)
    return bg

def get_random_background(size=(640, 640)):
    r = random.random()
    if r < 0.45: return create_marble_background(size)
    elif r < 0.70: return create_white_desk_background(size)
    elif r < 0.85: return create_dark_wood_background(size)
    else: return create_colored_mat_background(size)

def create_synthetic_booklet_spread(size=(460, 320)):
    w, h = size
    paper_bgr = (random.randint(235, 250), random.randint(235, 250), random.randint(235, 250))
    spread = np.full((h, w, 3), paper_bgr, dtype=np.uint8)
    mid_x = w // 2
    cv2.line(spread, (mid_x, 0), (mid_x, h), (180, 180, 180), 2)
    for dx in range(1, 15):
        alpha = (15 - dx) / 15.0 * 0.18
        spread[:, max(0, mid_x - dx)] = np.clip(spread[:, max(0, mid_x - dx)].astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
        spread[:, min(w - 1, mid_x + dx)] = np.clip(spread[:, min(w - 1, mid_x + dx)].astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
    line_spacing = random.randint(18, 24)
    line_col = (random.randint(150, 190), random.randint(140, 170), random.randint(120, 150))
    for y in range(40, h - 25, line_spacing):
        cv2.line(spread, (30, y), (mid_x - 15, y), line_col, 1)
        cv2.line(spread, (mid_x + 15, y), (w - 30, y), line_col, 1)
    margin_col = (random.randint(50, 80), random.randint(50, 80), random.randint(180, 220))
    cv2.line(spread, (55, 10), (55, h - 10), margin_col, 1)
    for y in range(40, h - 35, line_spacing):
        if random.random() < 0.7:
            cur_x = 65
            ink_col = (random.randint(110, 150), random.randint(40, 70), random.randint(20, 50))
            while cur_x < mid_x - 25:
                w_len = random.randint(20, 60)
                pts = [(cur_x + k * 8, y - random.randint(0, 6)) for k in range(w_len // 8)]
                for k in range(len(pts) - 1): cv2.line(spread, pts[k], pts[k + 1], ink_col, 1)
                cur_x += w_len + random.randint(8, 16)
    return spread

def composite_sample(bg, booklet, out_size=(640, 640)):
    bw, bh = booklet.shape[1], booklet.shape[0]
    out_w, out_h = out_size
    target_scale = random.uniform(0.65, 0.88) * min(out_w / bw, out_h / bh)
    new_w, new_h = int(bw * target_scale), int(bh * target_scale)
    resized_b = cv2.resize(booklet, (new_w, new_h))
    angle = random.uniform(-15.0, 15.0)
    rot_mat = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    cos, sin = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
    bound_w = int((new_h * sin) + (new_w * cos))
    bound_h = int((new_h * cos) + (new_w * sin))
    rot_mat[0, 2] += (bound_w / 2) - new_w / 2
    rot_mat[1, 2] += (bound_h / 2) - new_h / 2
    rotated_b = cv2.warpAffine(resized_b, rot_mat, (bound_w, bound_h), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((new_h, new_w), 255, dtype=np.uint8), rot_mat, (bound_w, bound_h), borderValue=0)
    max_x, max_y = max(10, out_w - bound_w - 10), max(10, out_h - bound_h - 10)
    px, py = random.randint(5, max_x), random.randint(5, max_y)
    roi = bg[py : py + bound_h, px : px + bound_w]
    mask_3d = (mask > 128)[:, :, None]
    shadow_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    shadow_alpha = (shadow_mask.astype(np.float32) / 255.0)[:, :, None] * 0.35
    roi[:] = np.clip(roi.astype(np.float32) * (1.0 - shadow_alpha), 0, 255).astype(np.uint8)
    roi[mask_3d.squeeze()] = rotated_b[mask_3d.squeeze()]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        bx, by, bw_box, bh_box = cv2.boundingRect(c)
        abs_x1, abs_y1 = px + bx, py + by
        abs_x2, abs_y2 = abs_x1 + bw_box, abs_y1 + bh_box
    else:
        abs_x1, abs_y1, abs_x2, abs_y2 = px, py, px + bound_w, py + bound_h
    cx = (abs_x1 + abs_x2) / 2.0 / out_w
    cy = (abs_y1 + abs_y2) / 2.0 / out_h
    norm_w = (abs_x2 - abs_x1) / float(out_w)
    norm_h = (abs_y2 - abs_y1) / float(out_h)
    return bg, (cx, cy, norm_w, norm_h)

def generate_augmented_dataset(project_root, num_train=120, num_val=30):
    train_img_dir = project_root / 'training' / 'dataset' / 'train' / 'images'
    train_lbl_dir = project_root / 'training' / 'dataset' / 'train' / 'labels'
    val_img_dir = project_root / 'training' / 'dataset' / 'valid' / 'images'
    val_lbl_dir = project_root / 'training' / 'dataset' / 'valid' / 'labels'
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    user_images = [
        (Path('C:/Users/Kavin/.gemini/antigravity/brain/82dc1272-af56-4ea3-9574-2010d5192c0e/.user_uploaded/media_1788983727559.jpg'), (0.5508, 0.4544, 0.8906, 0.8757), 'train'),
        (Path('C:/Users/Kavin/.gemini/antigravity/brain/82dc1272-af56-4ea3-9574-2010d5192c0e/.user_uploaded/media_1788983727580.jpg'), (0.5264, 0.4558, 0.9395, 0.8840), 'train'),
        (Path('C:/Users/Kavin/.gemini/antigravity/brain/82dc1272-af56-4ea3-9574-2010d5192c0e/.user_uploaded/media_1788983752935.jpg'), (0.5176, 0.3845, 0.6348, 0.7413), 'train'),
        (Path('C:/Users/Kavin/.gemini/antigravity/brain/82dc1272-af56-4ea3-9574-2010d5192c0e/.user_uploaded/media_1788983755372.jpg'), (0.5171, 0.4010, 0.6357, 0.7396), 'valid'),
    ]

    for idx, (p, (cx, cy, nw, nh), split) in enumerate(user_images):
        if p.exists():
            im = cv2.imread(str(p))
            fname = f'real_marble_user_{idx+1}.jpg'
            dest_img = (train_img_dir if split == 'train' else val_img_dir) / fname
            dest_lbl = (train_lbl_dir if split == 'train' else val_lbl_dir) / f'real_marble_user_{idx+1}.txt'
            dest_lbl.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            print(f'Added real sample: {fname} -> {split}')

    for i in range(num_train):
        bg = get_random_background((640, 640))
        booklet = create_synthetic_booklet_spread()
        comp, (cx, cy, nw, nh) = composite_sample(bg, booklet, (640, 640))
        fname = f'aug_bg_train_{i:04d}.jpg'
        cv2.imwrite(str(train_img_dir / fname), comp)
        (train_lbl_dir / f'aug_bg_train_{i:04d}.txt').write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    for i in range(num_val):
        bg = get_random_background((640, 640))
        booklet = create_synthetic_booklet_spread()
        comp, (cx, cy, nw, nh) = composite_sample(bg, booklet, (640, 640))
        fname = f'aug_bg_val_{i:04d}.jpg'
        cv2.imwrite(str(val_img_dir / fname), comp)
        (val_lbl_dir / f'aug_bg_val_{i:04d}.txt').write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f'Augmentation complete: {num_train} train + {num_val} val images created.')


if __name__ == '__main__':
    generate_augmented_dataset(Path('.'))
