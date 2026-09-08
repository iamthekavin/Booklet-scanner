import cv2
import os
import csv
import time
from datetime import datetime

# Output Configuration
# As requested, outputting to images/train to keep everything organized for YOLO
BASE_DIR = r"D:\VEE_SCAN\training\dataset\images\train"
LOG_FILE = r"D:\VEE_SCAN\training\dataset\capture_log.csv"

CATEGORIES = ["clean", "angle", "dist", "light", "hand", "blur"]
BLUR_THRESHOLD = 100.0  # Variance of Laplacian threshold (adjust if needed based on lighting/camera)

def get_next_sequence(category_dir):
    """Finds the highest sequence number in the directory and returns the next."""
    if not os.path.exists(category_dir):
        return 1
    files = [f for f in os.listdir(category_dir) if f.startswith("booklet_") and f.endswith(".jpg")]
    if not files:
        return 1
    seqs = []
    for f in files:
        try:
            # Expected format: booklet_category_001.jpg
            parts = f.split("_")
            seq = int(parts[-1].split(".")[0])
            seqs.append(seq)
        except Exception:
            pass
    return max(seqs) + 1 if seqs else 1

def calculate_sharpness(image):
    """Returns variance of Laplacian as a proxy for sharpness (blur detection)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    """Returns mean pixel value as a proxy for overall brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def main():
    print("=== VEE Scanner Capture Assistant ===")
    
    # 1. Prompt for category
    print("\nAvailable categories:")
    for i, cat in enumerate(CATEGORIES):
        print(f"{i + 1}. {cat}")
        
    while True:
        try:
            choice = input(f"\nSelect a category (1-{len(CATEGORIES)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(CATEGORIES):
                category = CATEGORIES[choice_idx]
                break
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a number.")
            
    # 2. Prompt for IP Webcam
    while True:
        ip_addr = input("\nEnter phone IP (e.g., 192.168.1.100): ").strip()
        if ip_addr:
            ip_webcam_url = f"http://{ip_addr}:8080/video"
            break
        print("IP address cannot be empty.")
        
    # Ensure directories exist
    category_dir = os.path.join(BASE_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    # 3. Setup log file
    log_exists = os.path.exists(LOG_FILE)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 4. Connect to camera
    print(f"\nConnecting to {ip_webcam_url} ...")
    cap = cv2.VideoCapture(ip_webcam_url)
    
    if not cap.isOpened():
        print("Failed to open the IP Webcam stream. Please check the IP and ensure the server is running.")
        return
        
    # 5. Initialize counters
    sequence = get_next_sequence(category_dir)
    capture_count = len([f for f in os.listdir(category_dir) if f.endswith(".jpg")]) if os.path.exists(category_dir) else 0
    
    print(f"\nReady to capture '{category}' images!")
    print(f"Target is ~15-30 images per category. You currently have {capture_count}.")
    print("\nCONTROLS:")
    print(" - SPACE : Capture a frame")
    print(" - Q / ESC : Quit")
    print("==================================\n")
    
    state = "LIVE"
    captured_frame = None
    captured_sharpness = 0.0
    captured_brightness = 0.0
    
    # Give the stream a second to buffer
    time.sleep(1)
    
    while True:
        if state == "LIVE":
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame. Retrying...")
                time.sleep(0.5)
                continue
                
            display_frame = frame.copy()
            
            # Draw HUD
            cv2.putText(display_frame, f"Cat: {category} | Count: {capture_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE to Capture | Q to Quit", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
            cv2.imshow("Capture Assistant", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                captured_frame = frame.copy()
                captured_sharpness = calculate_sharpness(captured_frame)
                captured_brightness = calculate_brightness(captured_frame)
                
                # Check blur unless category is intentionally blurry
                if captured_sharpness < BLUR_THRESHOLD and category != "blur":
                    state = "CONFIRM_BLUR"
                else:
                    state = "SAVE"
                    
        elif state == "CONFIRM_BLUR":
            display_frame = captured_frame.copy()
            
            cv2.putText(display_frame, f"WARNING: BLURRY (Score: {captured_sharpness:.1f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(display_frame, "Press 'S' to Save anyway | 'R' to Retake", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
            cv2.imshow("Capture Assistant", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                state = "SAVE"
            elif key == ord('r') or key == ord('R') or key == ord(' '):
                state = "LIVE"
                
        elif state == "SAVE":
            filename = f"booklet_{category}_{sequence:03d}.jpg"
            filepath = os.path.join(category_dir, filename)
            
            cv2.imwrite(filepath, captured_frame)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Log to CSV
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not log_exists:
                    writer.writerow(["filename", "category", "timestamp", "sharpness_score", "brightness_score"])
                    log_exists = True
                writer.writerow([filename, category, timestamp, f"{captured_sharpness:.2f}", f"{captured_brightness:.2f}"])
                
            print(f"Saved: {filename} | Sharpness: {captured_sharpness:.1f} | Brightness: {captured_brightness:.1f}")
            
            sequence += 1
            capture_count += 1
            state = "LIVE"
            
    cap.release()
    cv2.destroyAllWindows()
    print("\nCapture session ended.")

if __name__ == "__main__":
    main()
