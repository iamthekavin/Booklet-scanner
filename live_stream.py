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
SHARPNESS_MIN = 150.0
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
# Old OpenCV contour-only detection  (for comparison)
# ═══════════════════════════════════════════════════════════════════════

def opencv_contour_detect(frame: np.ndarray, min_area_ratio: float = 0.05) -> tuple[np.ndarray | None, int, float]:
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    num_pts = 0
    corners = None
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area_ratio * h * w:
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            num_pts = len(approx)
            if num_pts == 4:
                corners = CornerRefiner.order_corners(approx.reshape(4, 2).astype(np.float32))

    return corners, num_pts, (time.perf_counter() - t0) * 1000.0


# ═══════════════════════════════════════════════════════════════════════
# Drawing helpers
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

def draw_cv_geometry(vis: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    if corners is not None:
        pts = corners.astype(np.int32)
        for i in range(4):
            p1, p2 = tuple(pts[i]), tuple(pts[(i + 1) % 4])
            cv2.line(vis, p1, p2, MAGENTA, 2, cv2.LINE_AA)
            cv2.circle(vis, p1, 7, RED, -1, cv2.LINE_AA)
    return vis

def build_yolo_panel(vis_360: np.ndarray, result: DetectionResult, fps: float, frame_count: int, conf_thresh: float) -> np.ndarray:
    """Builds the 640x492 panel by vertically stacking the 40px top bar, 360px video, and 92px bottom bar."""
    # Central warning on video
    if ReviewFlag.NO_DETECTION in result.review_flags:
        text = "NO BOOKLET DETECTED"
        tsize = cv2.getTextSize(text, FONT, 0.8, 2)[0]
        cv2.putText(vis_360, text, ((640 - tsize[0]) // 2, 180), FONT, 0.8, RED, 2)
        
    # Top Bar (640 x 40)
    top_bar = np.full((40, 640, 3), DARK_BG, dtype=np.uint8)
    cv2.putText(top_bar, "YOLO11 Detection (NEW)", (10, 26), FONT, 0.7, GREEN, 2)
    cv2.putText(top_bar, f"FPS: {fps:.1f} | Frame: {frame_count}", (450, 16), FONT_SMALL, 0.9, YELLOW, 1)
    cv2.putText(top_bar, f"Conf >= {conf_thresh:.2f}", (450, 32), FONT_SMALL, 0.9, WHITE, 1)
    
    # Bottom Bar (640 x 92)
    bg_color = DARK_RED if result.needs_review else DARK_BG
    bot_bar = np.full((92, 640, 3), bg_color, dtype=np.uint8)
    
    c_val = f"{result.confidence:.3f}" if result.confidence > 0 else "0.000"
    cv2.putText(bot_bar, f"Method: {result.detection_method.value}", (10, 20), FONT_SMALL, 1.0, WHITE, 1)
    cv2.putText(bot_bar, f"Conf: {c_val} | Latency: {result.latency_ms:.0f}ms", (10, 40), FONT_SMALL, 1.0, ORANGE, 1)
    
    flags = [f.value for f in result.review_flags]
    if not flags: flags = ["none"]
    cv2.putText(bot_bar, f"Flags: [{','.join(flags)}]", (10, 60), FONT_SMALL, 1.0, ORANGE, 1)
    
    if result.needs_review:
        cv2.putText(bot_bar, "! NEEDS REVIEW - REPOSITION", (280, 45), FONT, 0.6, RED, 2)
        
    return np.vstack([top_bar, vis_360, bot_bar])

def build_cv_panel(vis_360: np.ndarray, corners: np.ndarray | None, num_pts: int, elapsed_ms: float) -> np.ndarray:
    """Builds the 640x492 panel by vertically stacking the 40px top bar, 360px video, and 92px bottom bar."""
    if corners is None:
        text = "FAILED"
        tsize = cv2.getTextSize(text, FONT, 0.8, 2)[0]
        cv2.putText(vis_360, text, ((640 - tsize[0]) // 2, 180), FONT, 0.8, RED, 2)
        
    # Top Bar (640 x 40)
    top_bar = np.full((40, 640, 3), DARK_BG, dtype=np.uint8)
    cv2.putText(top_bar, "OpenCV Contour (OLD)", (10, 26), FONT, 0.7, RED, 2)
    
    # Bottom Bar (640 x 92)
    bot_bar = np.full((92, 640, 3), DARK_BG, dtype=np.uint8)
    if corners is not None:
        cv2.putText(bot_bar, f"4-pt contour ({elapsed_ms:.1f}ms)", (10, 20), FONT_SMALL, 1.0, GREEN, 1)
    else:
        cv2.putText(bot_bar, f"No 4-pt fit ({num_pts} pts, {elapsed_ms:.0f}ms)", (10, 20), FONT_SMALL, 1.0, YELLOW, 1)
        
    return np.vstack([top_bar, vis_360, bot_bar])

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
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
            if not ret:
                time.sleep(0.01)

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
    parser.add_argument("--ip", type=str, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--webcam", type=int, default=None)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_dir   = project_root / "output"
    dataset_dir  = project_root / "training" / "dataset"

    print("\n" + "=" * 62)
    print("  VEE Scanner — Live Booklet Detection")
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

    session_dir = output_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    session = BookletCaptureSession(session_dir=session_dir, warper=warper)

    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    WINDOW_NAME = "VEE Scanner - Dual Panel"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print("  Stream connected.")
    print("  Controls:")
    print("    S : Capture spread -> split into 2 pages (001.jpg, 002.jpg) -> compile to PDF (FR-4.2)")
    print("    W : Toggle warped booklet window")
    print("    C : Capture training frame")
    print("    +/- : Adjust confidence threshold")
    print("    Q : Quit\n")

    try:
        while True:
            t_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            # Quality Gate
            q_pass, q_sharpness, q_glare, q_reason = check_image_quality(frame)

            # 1. Detect YOLO
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

                warped = warper.warp_adaptive(frame, result.corners.as_float32()) if result.corners is not None else None
                yolo_vis = draw_yolo_geometry(frame.copy(), result)
            else:
                result = DetectionResult(
                    bbox=None, corners=None, confidence=0.0,
                    detection_method=DetectionMethod.NONE,
                    review_flags=[ReviewFlag.NO_DETECTION],
                    needs_review=False, hands_detected=[], clutter_detected=[],
                    latency_ms=0.0, raw_detections=[], frame_shape=frame.shape[:2]
                )
                warped = None
                yolo_vis = frame.copy()

            # 2. Detect OpenCV
            if q_pass:
                cv_corners, cv_pts, cv_ms = opencv_contour_detect(frame)
                cv_vis = draw_cv_geometry(frame.copy(), cv_corners)
            else:
                cv_corners, cv_pts, cv_ms = None, 0, 0.0
                cv_vis = frame.copy()

            # 3. Resize videos to proper 16:9 ratio (640x360) BEFORE adding headers/footers
            yp_360 = cv2.resize(yolo_vis, (640, 360))
            cp_360 = cv2.resize(cv_vis, (640, 360))
            
            # Add Quality Gate overlay to the YOLO panel
            q_color = GREEN if q_pass else RED
            cv2.putText(yp_360, f"Gate: {q_reason} | Sharp: {q_sharpness:.0f} | Glare: {q_glare:.1%}", (10, 25), FONT_SMALL, 1.2, q_color, 2)

            # 4. Build Panels (stacks 40px header + 360px video + 92px footer = 492px)
            fps = 1.0 / (sum(fps_history) / len(fps_history)) if fps_history else 0.0
            yp = build_yolo_panel(yp_360, result, fps, frame_count, conf_thresh)
            cp = build_cv_panel(cp_360, cv_corners, cv_pts, cv_ms)

            # 5. Stack horizontally -> exact 1280x492 base layout
            display = np.hstack([yp, cp])
            
            # 6. Scale up the final image to make the window physically larger on screen
            if args.scale != 1.0:
                display = cv2.resize(display, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)

            elapsed = time.perf_counter() - t_frame
            fps_history.append(elapsed)

            cv2.imshow(WINDOW_NAME, display)

            if show_warp and warped is not None:
                warp_disp = warped.copy()
                scale = min(600 / warp_disp.shape[0], 500 / warp_disp.shape[1])
                if scale < 1:
                    warp_disp = cv2.resize(warp_disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("Warped Booklet", warp_disp)
            elif show_warp and warped is None:
                blank = np.full((200, 400, 3), 40, dtype=np.uint8)
                cv2.putText(blank, "No booklet detected", (40, 110), FONT, 0.7, RED, 2)
                cv2.imshow("Warped Booklet", blank)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27): break
            elif key in (ord('w'), ord('W')):
                show_warp = not show_warp
                if not show_warp: cv2.destroyWindow("Warped Booklet")
            elif key in (ord('s'), ord('S')):
                if warped is not None and result.corners is not None:
                    # Attempt high-res capture
                    high_res_frame = capture_high_res_frame(args)
                    if high_res_frame is not None:
                        # Re-run detection on the high-res frame to ensure perfect crop 
                        # regardless of camera movement during the HTTP fetch or FOV differences.
                        # This runs YOLO exactly the same way, just on the new image.
                        hr_result = detector.detect(high_res_frame)
                        if hr_result.corners is not None:
                            warped_to_save = warper.warp_adaptive(high_res_frame, hr_result.corners.as_float32())
                        else:
                            # Fallback to scaling preview corners if detection fails
                            h_orig, w_orig = frame.shape[:2]
                            h_high, w_high = high_res_frame.shape[:2]
                            scale_x = w_high / w_orig
                            scale_y = h_high / h_orig
                            corners_scaled = result.corners.as_float32().copy()
                            corners_scaled[:, 0] *= scale_x
                            corners_scaled[:, 1] *= scale_y
                            warped_to_save = warper.warp_adaptive(high_res_frame, corners_scaled)
                            
                        raw_to_save = high_res_frame
                        print(f"\n  ✨ High-res capture successful: {high_res_frame.shape[1]}x{high_res_frame.shape[0]}")
                    else:
                        warped_to_save = warped
                        raw_to_save = frame
                        
                    saved_pages = session.add_spread(warped_to_save, raw_frame=raw_to_save)
                    pdf_path = session.compile_pdf()
                    if len(saved_pages) == 2:
                        print(f"  📸 Spread #{session.spread_count} captured & split into 2 pages (FR-4.2):")
                        print(f"     ├── Left page:  {saved_pages[0].name}")
                        print(f"     └── Right page: {saved_pages[1].name}")
                    else:
                        print(f"  📸 Cover / Single page #{session.spread_count} captured:")
                        print(f"     └── Saved:      {saved_pages[0].name}")
                    print(f"  📄 PDF compiled: {pdf_path.name} (Total pages: {session.page_count})\n")
                else:
                    saved = save_frame(frame, yolo_vis, warped, output_dir)
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
