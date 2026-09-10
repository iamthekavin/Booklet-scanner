"""
VEE Scanner — Live IP Webcam Booklet Detection (Restored UI Layout)
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from collections import deque
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure local imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import BookletDetector
from config import DetectorConfig
from corner_refiner import CornerRefiner
from models import DetectionMethod, DetectionResult, ReviewFlag
from perspective import PerspectiveWarper
from pdf_compiler import BookletCaptureSession, split_spread_to_pages, save_split_pages, compile_pages_to_pdf
from auto_capture import AutoCaptureController, AutoCaptureConfig, AutoCaptureState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_detect")

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Colours (BGR)
GREEN       = (0, 255, 0)
RED         = (0, 0, 255)
CYAN        = (255, 255, 0)
YELLOW      = (0, 255, 255)
ORANGE      = (0, 165, 255)
WHITE       = (255, 255, 255)
DARK_BG     = (30, 30, 30)
DARK_RED    = (0, 0, 100)
MAGENTA     = (255, 0, 255)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_PLAIN

# ═══════════════════════════════════════════════════════════════════════
# Quality Gate Constants
# ═══════════════════════════════════════════════════════════════════════
SHARPNESS_MIN = 45.0
GLARE_FRAC_MAX = 0.02

def check_image_quality(frame: np.ndarray) -> tuple[bool, float, float, str]:
    """
    Checks frame for sharpness and glare before detection.
    Returns: (passed, sharpness, glare_frac, reason)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sharpness: variance of the Laplacian
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Glare: fraction of saturated/overexposed pixels (>250)
    # Using threshold to find pixels > 250
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    glare_pixels = cv2.countNonZero(thresh)
    glare_frac = glare_pixels / (gray.shape[0] * gray.shape[1])
    
    if sharpness < SHARPNESS_MIN:
        return False, sharpness, glare_frac, "BLURRY"
    if glare_frac > GLARE_FRAC_MAX:
        return False, sharpness, glare_frac, "GLARE"
        
    return True, sharpness, glare_frac, "OK"

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# Drawing & UI Helpers (YOLO Single-Panel)
# ═══════════════════════════════════════════════════════════════════════

def draw_yolo_geometry(vis: np.ndarray, result: DetectionResult) -> np.ndarray:
    if result.bbox is not None:
        x1, y1 = int(result.bbox.x1), int(result.bbox.y1)
        x2, y2 = int(result.bbox.x2), int(result.bbox.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN, 2)

    if result.corners is not None:
        pts = result.corners.points.astype(np.int32)
        for i in range(4):
            p1, p2 = tuple(pts[i]), tuple(pts[(i + 1) % 4])
            cv2.line(vis, p1, p2, CYAN, 2, cv2.LINE_AA)
            cv2.circle(vis, p1, 7, RED, -1, cv2.LINE_AA)
            cv2.putText(vis, ["TL", "TR", "BR", "BL"][i], (p1[0] + 10, p1[1] - 10), FONT_SMALL, 1.2, WHITE, 1)

    for hand in result.hands_detected:
        hx1, hy1, hx2, hy2 = int(hand.x1), int(hand.y1), int(hand.x2), int(hand.y2)
        cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), ORANGE, 2)
        cv2.putText(vis, f"hand {hand.confidence:.0%}", (hx1, hy1 - 6), FONT_SMALL, 1.1, ORANGE, 1)
        
    return vis


def build_scanner_panel(
    vis: np.ndarray,
    result: DetectionResult,
    fps: float,
    frame_count: int,
    conf_thresh: float,
    auto_enabled: bool = True,
    top_h: int = 42,
    bot_h: int = 76,
) -> np.ndarray:
    """Builds the single-panel UI by vertically stacking top header, video, and bottom footer."""
    h, w = vis.shape[:2]

    # Central warning on video if no booklet detected
    if ReviewFlag.NO_DETECTION in result.review_flags:
        text = "NO BOOKLET DETECTED"
        scale = 1.0 if w >= 1280 else 0.8
        tsize = cv2.getTextSize(text, FONT, scale, 2)[0]
        cv2.putText(vis, text, ((w - tsize[0]) // 2, h // 2), FONT, scale, RED, 2)

    # Top Bar (w x top_h)
    top_bar = np.full((top_h, w, 3), DARK_BG, dtype=np.uint8)
    title_scale = 0.8 if w >= 1280 else 0.7
    cv2.putText(top_bar, "VEE Scanner — YOLO11 Detection", (16, int(top_h * 0.68)), FONT, title_scale, GREEN, 2)

    # Auto-Capture toggle indicator in center
    auto_label = "[A] Auto-Capture: ON" if auto_enabled else "[A] Auto-Capture: OFF"
    auto_col = GREEN if auto_enabled else (120, 120, 120)
    auto_scale = 1.2 if w >= 1280 else 1.1
    auto_size = cv2.getTextSize(auto_label, FONT_SMALL, auto_scale, 1)[0]
    cv2.putText(top_bar, auto_label, ((w - auto_size[0]) // 2, int(top_h * 0.68)), FONT_SMALL, auto_scale, auto_col, 1)

    # Stats on right
    stats_txt = f"FPS: {fps:.1f} | Frame: {frame_count} | Conf >= {conf_thresh:.2f}"
    stats_scale = 1.05 if w >= 1280 else 0.95
    stats_size = cv2.getTextSize(stats_txt, FONT_SMALL, stats_scale, 1)[0]
    cv2.putText(top_bar, stats_txt, (w - stats_size[0] - 16, int(top_h * 0.68)), FONT_SMALL, stats_scale, YELLOW, 1)

    # Bottom Bar (w x bot_h)
    bg_color = DARK_RED if result.needs_review else DARK_BG
    bot_bar = np.full((bot_h, w, 3), bg_color, dtype=np.uint8)

    c_val = f"{result.confidence:.3f}" if result.confidence > 0 else "0.000"
    flags = [f.value for f in result.review_flags]
    if not flags:
        flags = ["ok"]

    line1 = f"Method: {result.detection_method.value}  |  Conf: {c_val}  |  Latency: {result.latency_ms:.0f}ms  |  Flags: [{','.join(flags)}]"
    cv2.putText(bot_bar, line1, (16, int(bot_h * 0.36)), FONT_SMALL, 1.15 if w >= 1280 else 1.0, WHITE, 1)

    if result.needs_review:
        cv2.putText(bot_bar, "! NEEDS REVIEW - REPOSITION BOOKLET", (16, int(bot_h * 0.78)), FONT, 0.7, RED, 2)
    else:
        controls_txt = "[A] Auto ON/OFF  |  [S] Manual Capture  |  [F] Fullscreen  |  [W] Warped View  |  [C] Train Snap  |  [+/-] Conf  |  [Q] Quit"
        cv2.putText(bot_bar, controls_txt, (16, int(bot_h * 0.78)), FONT_SMALL, 1.05 if w >= 1280 else 0.95, (190, 190, 190), 1)

    return np.vstack([top_bar, vis, bot_bar])

# ═══════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════

class ThreadedCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            # Drain internal stream buffer with grab() to guarantee zero lag/latency
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            ret, frame = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

def capture_high_res_frame(args: argparse.Namespace) -> np.ndarray | None:
    """Attempts to capture a full-resolution snapshot from IP Webcam's /shot.jpg endpoint."""
    if args.webcam is not None:
        return None
        
    shot_url = None
    if args.url and "video" in args.url:
        shot_url = args.url.replace("video", "shot.jpg")
    elif not args.url:
        ip = args.ip or "192.168.1.10"
        shot_url = f"http://{ip}:8080/shot.jpg"
        
    if shot_url:
        import urllib.request
        try:
            logger.info(f"Requesting high-res frame from {shot_url}")
            # Request high quality JPEG
            req = urllib.request.Request(shot_url)
            with urllib.request.urlopen(req, timeout=3) as response:
                img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Failed to fetch high-res frame from {shot_url}: {e}")
            
    return None

def open_stream(args: argparse.Namespace) -> ThreadedCamera:
    if getattr(args, "source", None) is not None:
        src_str = str(args.source).strip()
        if src_str.isdigit():
            args.webcam = int(src_str)
        elif src_str.startswith("http://") or src_str.startswith("https://") or src_str.startswith("rtsp://"):
            args.url = src_str
        else:
            args.ip = src_str

    if args.webcam is not None:
        source = int(args.webcam)
        logger.info(f"Opening local webcam index {source}")
    elif args.url:
        source = args.url
        logger.info(f"Opening stream URL: {source}")
    else:
        ip = args.ip or "192.168.1.10"
        source = f"http://{ip}:8080/video"
        logger.info(f"Opening IP Webcam stream: {source}")

    cap = ThreadedCamera(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        print(f"\n❌ Cannot connect to: {source}")
        sys.exit(1)

    logger.info("Stream opened successfully")
    return cap

def play_shutter_sound() -> None:
    """Plays a non-blocking camera shutter beep on Windows."""
    try:
        import winsound
        winsound.Beep(1400, 90)
        winsound.Beep(1800, 70)
    except Exception:
        pass


def draw_auto_capture_hud(
    vis: np.ndarray,
    state: AutoCaptureState,
    progress: float,
    status_msg: str,
    flash_active: bool,
    page_count: int,
    booklet_idx: int = 1,
    frames_per_booklet: Optional[int] = None,
    spread_count: int = 0,
) -> np.ndarray:
    """Draws auto-capture progress bar, status, and capture flash HUD."""
    h, w = vis.shape[:2]

    # Flash banner on capture
    if flash_active:
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), GREEN, 8)
        banner_w, banner_h = min(560, int(w * 0.52)), 52
        bx1 = (w - banner_w) // 2
        by1 = 45
        cv2.rectangle(vis, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 140, 0), -1)
        cv2.rectangle(vis, (bx1, by1), (bx1 + banner_w, by1 + banner_h), WHITE, 2)
        if frames_per_booklet:
            text = f"CAPTURED! B#{booklet_idx:02d} F{spread_count}/{frames_per_booklet} (P#{page_count})"
        else:
            text = f"CAPTURED! PAGE #{page_count}"
        tsize = cv2.getTextSize(text, FONT, 0.75, 2)[0]
        cv2.putText(vis, text, (bx1 + (banner_w - tsize[0]) // 2, by1 + 34), FONT, 0.75, WHITE, 2)
        return vis

    if frames_per_booklet is not None:
        b_tag = f"Booklet #{booklet_idx:02d} | Spread {spread_count}/{frames_per_booklet}"
        tsize = cv2.getTextSize(b_tag, FONT_SMALL, 1.1, 1)[0]
        cv2.putText(vis, b_tag, (w - tsize[0] - 16, 28), FONT_SMALL, 1.1, CYAN, 1)

    if state == AutoCaptureState.STABILIZING:
        bar_w = min(480, int(w * 0.45))
        bar_x = (w - bar_w) // 2
        bar_y = h - 42
        bar_h = 28
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), DARK_BG, -1)
        fill_w = int(max(0.0, min(1.0, progress)) * (bar_w - 4))
        fill_color = GREEN if progress >= 0.8 else CYAN
        if fill_w > 0:
            cv2.rectangle(vis, (bar_x + 2, bar_y + 2), (bar_x + 2 + fill_w, bar_y + bar_h - 2), fill_color, -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)
        pct = int(progress * 100)
        label = f"Auto-Capture: {pct}%"
        tsize = cv2.getTextSize(label, FONT_SMALL, 1.1, 1)[0]
        cv2.putText(vis, label, (bar_x + (bar_w - tsize[0]) // 2, bar_y + 19), FONT_SMALL, 1.1, WHITE, 1)

    elif state == AutoCaptureState.WAITING_FOR_PAGE_TURN:
        bar_w = min(400, int(w * 0.4))
        bar_x = (w - bar_w) // 2
        bar_y = h - 42
        bar_h = 28
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 80), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), YELLOW, 1)
        label = "Turn to next page..."
        tsize = cv2.getTextSize(label, FONT_SMALL, 1.1, 1)[0]
        cv2.putText(vis, label, (bar_x + (bar_w - tsize[0]) // 2, bar_y + 19), FONT_SMALL, 1.1, YELLOW, 1)

    return vis


def execute_capture(
    frame: np.ndarray,
    result: DetectionResult,
    args: argparse.Namespace,
    detector: BookletDetector,
    warper: PerspectiveWarper,
    session: BookletCaptureSession,
    auto_controller: AutoCaptureController,
    is_auto: bool = False,
    cached_warped: Optional[np.ndarray] = None,
) -> bool:
    """Executes spread capture, high-res fallback, warp, PDF update, and shutter audio."""
    if result.corners is None:
        return False

    high_res_frame = capture_high_res_frame(args)
    if high_res_frame is not None:
        hr_result = detector.detect(high_res_frame)
        if hr_result.corners is not None:
            warped_to_save = warper.warp_adaptive(high_res_frame, hr_result.corners.as_float32(), high_quality=True)
        else:
            h_orig, w_orig = frame.shape[:2]
            h_high, w_high = high_res_frame.shape[:2]
            scale_x = w_high / w_orig
            scale_y = h_high / h_orig
            corners_scaled = result.corners.as_float32().copy()
            corners_scaled[:, 0] *= scale_x
            corners_scaled[:, 1] *= scale_y
            warped_to_save = warper.warp_adaptive(high_res_frame, corners_scaled, high_quality=True)
        raw_to_save = high_res_frame
        print(f"\n  ✨ High-res capture successful: {high_res_frame.shape[1]}x{high_res_frame.shape[0]}")
    else:
        if cached_warped is not None:
            warped_to_save = cached_warped
        else:
            warped_to_save = warper.warp_adaptive(frame, result.corners.as_float32(), high_quality=True)
        raw_to_save = frame

    if warped_to_save is None:
        return False

    saved_pages = session.add_spread(warped_to_save, raw_frame=raw_to_save)
    tag = "🤖 Auto-Capture" if is_auto else "📸 Manual Capture"
    if not saved_pages:
        print(f"\n  ⚠️ {tag} skipped: duplicate spread detected (pHash distance <= {session.dup_hash_dist})!")
        return False

    pdf_path = session.compile_pdf()

    auto_controller.notify_manual_capture(result.corners.as_float32())
    threading.Thread(target=play_shutter_sound, daemon=True).start()

    if len(saved_pages) == 2:
        print(f"\n  {tag}: Spread #{session.spread_count} captured & split into 2 pages (FR-4.2):")
        print(f"     ├── Left page:  {saved_pages[0].name}")
        print(f"     └── Right page: {saved_pages[1].name}")
    else:
        print(f"\n  {tag}: Cover / Single page #{session.spread_count} captured:")
        print(f"     └── Saved:      {saved_pages[0].name}")
    print(f"  📄 PDF compiled: {pdf_path.name} (Total pages: {session.page_count})\n")

    if session.is_booklet_complete():
        print("\n" + "=" * 62)
        print(f"  🎉 BOOKLET #{session.booklet_idx:02d} COMPLETED! ({session.page_count} pages)")
        print(f"  📁 PDF finalized: {pdf_path}")
        print("=" * 62)
        session.roll_over_to_next_booklet()
        auto_controller.reset()
        print(f"  ✨ Ready for Booklet #{session.booklet_idx:02d} — place cover page to begin.\n")

    return True



def save_frame(frame, annotated, warped, output_dir, prefix="snap"):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{prefix}_{ts}"
    cv2.imwrite(str(base.with_suffix(".jpg")), frame)
    cv2.imwrite(str(base.with_name(base.name + "_det.jpg")), annotated)
    if warped is not None:
        cv2.imwrite(str(base.with_name(base.name + "_warp.jpg")), warped)
    logger.info(f"Saved to {base}*")
    return base

def capture_training_frame(frame, dataset_dir):
    img_dir = dataset_dir / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    idx = len(list(img_dir.glob("*.jpg"))) + 1
    name = f"booklet_{idx:04d}_{ts}.jpg"
    cv2.imwrite(str(img_dir / name), frame)
    print(f"  📸 Training image saved: {name}  (total: {idx})")

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, default=None, help="Video source (webcam index like 0, IP address, or stream URL)")
    parser.add_argument("--ip", type=str, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--webcam", type=int, default=None)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0, help="Display scaling factor (default: 1.0)")
    parser.add_argument("--fullscreen", action="store_true", default=False, help="Launch directly in true borderless fullscreen")
    parser.add_argument("--res", type=str, default="1280x720", choices=["1280x720", "1920x1080"], help="Base UI canvas resolution (default: 1280x720)")
    parser.add_argument("--auto", "--auto-capture", dest="auto_capture", action="store_true", default=True, help="Enable auto-capture (default: True)")
    parser.add_argument("--no-auto", dest="auto_capture", action="store_false", help="Disable auto-capture")
    parser.add_argument("--auto-delay", type=float, default=1.0, help="Hold duration in seconds for auto-capture (default: 1.0s)")
    parser.add_argument("--booklet-frames", type=int, default=None, help="Spreads per booklet for auto-roll (e.g. 8 for an 8-spread / 16-page booklet)")
    parser.add_argument("--dup-hash-dist", type=int, default=6, help="pHash Hamming distance threshold to reject duplicates (default: 6)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_dir   = project_root / "output"
    dataset_dir  = project_root / "training" / "dataset"

    print("\n" + "=" * 62)
    print("  VEE Scanner — Live Booklet Detection (YOLO11)")
    print("=" * 62)

    config = DetectorConfig(model_path=args.model, device=args.device, confidence_threshold=args.conf)
    detector = BookletDetector(config)
    warper   = PerspectiveWarper()
    
    cap = open_stream(args)

    show_warp    = False            
    conf_thresh  = args.conf
    fps_history  = deque(maxlen=60)
    frame_count  = 0
    
    # Temporal smoothing for live preview (FR-2.4 stabilization)
    prev_corners = None
    ema_alpha = 0.3  # Smoothing factor: lower = smoother, higher = more responsive

    # Auto-Capture controller & visual flash
    auto_config = AutoCaptureConfig(stability_duration=args.auto_delay)
    auto_controller = AutoCaptureController(config=auto_config, enabled=args.auto_capture)
    flash_timer = 0.0

    session_dir = output_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    session = BookletCaptureSession(
        session_dir=session_dir,
        warper=warper,
        frames_per_booklet=args.booklet_frames,
        dup_hash_dist=args.dup_hash_dist,
    )

    WINDOW_NAME = "VEE Scanner — YOLO11 Live Detection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Determine screen resolution and maximize window to full window size
    screen_w, screen_h = 1920, 1080
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            pass

    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)
    is_fullscreen = bool(args.fullscreen)
    if is_fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("  Stream connected.")
    print("  Controls:")
    print("    A : Toggle Auto-Capture ON/OFF (hands-free scanning)")
    print("    S : Manual capture spread -> split & compile to PDF (FR-4.2)")
    print("    F : Toggle Fullscreen / Maximized window")
    print("    W : Toggle warped booklet window")
    print("    C : Capture training frame")
    print("    +/- : Adjust confidence threshold")
    print("    Q : Quit\n")
    print(f"  🤖 Auto-Capture is {'ON (hold steady 1.0s to capture)' if auto_controller.enabled else 'OFF'}\n")

    try:
        while True:
            t_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Quality Gate
            q_pass, q_sharpness, q_glare, q_reason = check_image_quality(frame)

            # 1. Detect YOLO + Corner Refinement
            if q_pass:
                result = detector.detect(frame)
                
                # Apply EMA Temporal Smoothing to eliminate jitter
                if result.corners is not None:
                    curr_pts = result.corners.points.astype(np.float32)
                    if prev_corners is not None:
                        # Reset EMA if jump is huge (>10% frame width), meaning new page or fast movement
                        if np.max(np.abs(curr_pts - prev_corners)) > frame.shape[1] * 0.1:
                            prev_corners = curr_pts
                        else:
                            curr_pts = ema_alpha * curr_pts + (1.0 - ema_alpha) * prev_corners
                            prev_corners = curr_pts
                    else:
                        prev_corners = curr_pts
                        
                    result.corners.points = curr_pts
                else:
                    prev_corners = None

                yolo_vis = draw_yolo_geometry(frame.copy(), result)
            else:
                result = DetectionResult(
                    bbox=None, corners=None, confidence=0.0,
                    detection_method=DetectionMethod.NONE,
                    review_flags=[ReviewFlag.NO_DETECTION],
                    needs_review=False, hands_detected=[], clutter_detected=[],
                    latency_ms=0.0, raw_detections=[], frame_shape=frame.shape[:2]
                )
                yolo_vis = frame.copy()

            # 2. Auto-Capture evaluation
            auto_state = AutoCaptureState.DISABLED
            auto_prog = 0.0
            auto_msg = "Auto: OFF [A]"
            if auto_controller.enabled:
                dt_step = elapsed if ('elapsed' in locals() and elapsed > 0) else 0.033
                should_auto_capture, auto_state, auto_prog, auto_msg = auto_controller.update(
                    result, q_pass=q_pass, dt=dt_step
                )
                if should_auto_capture and result.corners is not None:
                    flash_timer = 1.0
                    execute_capture(
                        frame, result, args, detector, warper, session, auto_controller, is_auto=True
                    )

            # 3. Resize video to fit 16:9 canvas (1280x720 or 1920x1080)
            if args.res == "1920x1080":
                canvas_w, canvas_h = 1920, 1080
                top_h, bot_h = 50, 84
            else:
                canvas_w, canvas_h = 1280, 720
                top_h, bot_h = 42, 76
            video_h = canvas_h - top_h - bot_h
            yp_vis = cv2.resize(yolo_vis, (canvas_w, video_h), interpolation=cv2.INTER_LINEAR)
            
            # Draw Auto-Capture HUD (progress bar / countdown / flash banner)
            draw_auto_capture_hud(
                yp_vis,
                auto_state,
                auto_prog,
                auto_msg,
                flash_active=(flash_timer > 0),
                page_count=session.page_count,
                booklet_idx=session.booklet_idx,
                frames_per_booklet=session.frames_per_booklet,
                spread_count=session.spread_count,
            )
            if 'elapsed' in locals() and elapsed > 0:
                flash_timer = max(0.0, flash_timer - elapsed)

            # Add Quality Gate overlay
            q_color = GREEN if q_pass else RED
            cv2.putText(yp_vis, f"Gate: {q_reason} | Sharp: {q_sharpness:.0f} | Glare: {q_glare:.1%}", (16, 26), FONT_SMALL, 1.2, q_color, 2)

            # 4. Build single-panel interface (header + video + footer)
            fps = 1.0 / (sum(fps_history) / len(fps_history)) if fps_history else 0.0
            display = build_scanner_panel(yp_vis, result, fps, frame_count, conf_thresh, auto_enabled=auto_controller.enabled, top_h=top_h, bot_h=bot_h)

            # 5. Optional display scaling
            if args.scale != 1.0:
                display = cv2.resize(display, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)

            elapsed = time.perf_counter() - t_frame
            fps_history.append(elapsed)

            cv2.imshow(WINDOW_NAME, display)

            # Auto-maximize window on Windows on first frame
            if frame_count == 1 and not is_fullscreen and os.name == 'nt':
                try:
                    import ctypes
                    hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
                    if hwnd:
                        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
                except Exception:
                    pass

            # Optional warped preview window (fast linear interpolation, on-demand only)
            if show_warp and result.corners is not None:
                warped_preview = warper.warp_adaptive(frame, result.corners.as_float32(), high_quality=False)
                if warped_preview is not None:
                    warp_disp = warped_preview.copy()
                    scale = min(600 / warp_disp.shape[0], 500 / warp_disp.shape[1])
                    if scale < 1:
                        warp_disp = cv2.resize(warp_disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("Warped Booklet", warp_disp)
            elif show_warp and result.corners is None:
                blank = np.full((200, 400, 3), 40, dtype=np.uint8)
                cv2.putText(blank, "No booklet detected", (40, 110), FONT, 0.7, RED, 2)
                cv2.imshow("Warped Booklet", blank)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27): break
            elif key in (ord('a'), ord('A')):
                enabled = auto_controller.toggle()
                print(f"\n  🤖 Auto-Capture: {'ENABLED' if enabled else 'DISABLED'} (Press 'A' to toggle)\n")
            elif key in (ord('f'), ord('F')):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("\n  📺 Fullscreen: ON (Press 'F' to toggle)\n")
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    if os.name == 'nt':
                        try:
                            import ctypes
                            hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
                            if hwnd:
                                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
                        except Exception:
                            pass
                    print("\n  🪟 Fullscreen: OFF (Maximized window)\n")
            elif key in (ord('w'), ord('W')):
                show_warp = not show_warp
                if not show_warp: cv2.destroyWindow("Warped Booklet")
            elif key in (ord('s'), ord('S')):
                if result.corners is not None:
                    flash_timer = 1.0
                    execute_capture(
                        frame, result, args, detector, warper, session, auto_controller, is_auto=False
                    )
                else:
                    saved = save_frame(frame, yolo_vis, None, output_dir)
                    print(f"\n  ⚠️ No booklet detected to split. Saved raw snapshot: {saved.name}.jpg\n")
            elif key in (ord('c'), ord('C')): capture_training_frame(frame, dataset_dir)
            elif key in (ord('+'), ord('=')):
                conf_thresh = min(conf_thresh + 0.05, 0.95)
                detector.config.confidence_threshold = conf_thresh
            elif key in (ord('-'), ord('_')):
                conf_thresh = max(conf_thresh - 0.05, 0.05)
                detector.config.confidence_threshold = conf_thresh

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if session.page_count > 0:
            print(f"  📕 Final session PDF: {session.session_dir / 'booklet_scan.pdf'} ({session.page_count} pages from {session.spread_count} spreads)")
        print(f"\n  Processed {frame_count} frames.  Goodbye.\n")

if __name__ == "__main__":
    main()
