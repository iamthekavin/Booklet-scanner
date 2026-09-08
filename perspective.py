"""
Perspective transformation module for VEE Scanner.
Handles perspective correction and page splitting of detected booklets.
"""

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from models import QuadCorners

logger = logging.getLogger(__name__)

# Scanner Configuration Constants
TARGET_DPI = 300
PAGE_WIDTH_MM = 210  # A4 width
PAGE_HEIGHT_MM = 297 # A4 height

def mm_to_px(mm: float, dpi: int) -> int:
    return int((mm / 25.4) * dpi)

class PerspectiveWarper:
    """Handles perspective transformation of booklet images."""

    def __init__(self, dpi: int = TARGET_DPI) -> None:
        self.dpi = dpi
        self.page_w_px = mm_to_px(PAGE_WIDTH_MM, self.dpi)
        self.page_h_px = mm_to_px(PAGE_HEIGHT_MM, self.dpi)

    def warp_adaptive(self, frame: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Warp the frame to a fixed physical scanner target (A4 or A3 spread) at configured DPI.
        """
        try:
            tl, tr, br, bl = corners

            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            avg_width = (width_top + width_bottom) / 2.0

            height_left = np.linalg.norm(tl - bl)
            height_right = np.linalg.norm(tr - br)
            avg_height = (height_left + height_right) / 2.0

            is_spread = avg_width > avg_height
            
            target_w = self.page_w_px * 2 if is_spread else self.page_w_px
            target_h = self.page_h_px

            dst_pts = np.array([
                [0, 0],
                [target_w, 0],
                [target_w, target_h],
                [0, target_h]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
            warped = cv2.warpPerspective(frame, matrix, (target_w, target_h))
            return warped
        except Exception as e:
            logger.error(f"Failed to adaptive warp image: {e}")
            return None

    def estimate_page_split(self, warped_image: np.ndarray, max_deviation_ratio: float = 0.03) -> int:
        """Find the vertical center fold / spine line in a warped booklet image."""
        try:
            if len(warped_image.shape) == 3:
                gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = warped_image

            height, width = gray.shape
            center_x = width // 2

            search_width = max(1, int(width * max_deviation_ratio))
            start_x = max(0, center_x - search_width)
            end_x = min(width, center_x + search_width)

            projection = np.sum(gray[:, start_x:end_x], axis=0, dtype=np.float32)

            if len(projection) >= 5:
                kernel = np.ones(5) / 5.0
                smoothed = np.convolve(projection, kernel, mode="same")
            else:
                smoothed = projection

            idx = int(np.argmin(smoothed))

            if 2 <= idx <= len(smoothed) - 3:
                left_edge = smoothed[0]
                right_edge = smoothed[-1]
                val = smoothed[idx]
                trough_depth = min(left_edge, right_edge) - val
                if trough_depth > 0.03 * min(left_edge, right_edge):
                    return start_x + idx

            return center_x
        except Exception as e:
            logger.error(f"Failed to estimate page split: {e}")
            return warped_image.shape[1] // 2

    def split_pages(self, warped_image: np.ndarray, spine_x: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Split the warped image into left and right pages at the spine."""
        if spine_x is None:
            spine_x = self.estimate_page_split(warped_image)
            
        left_page = warped_image[:, :spine_x]
        right_page = warped_image[:, spine_x:]
        
        return left_page, right_page

    def enhance_scan(self, page_image: np.ndarray) -> np.ndarray:
        """
        Enhancement and deskew (Part 3) removed per user request.
        """
        return page_image

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Classic 4-point perspective transform function.

    Args:
        image (np.ndarray): Source image.
        pts (np.ndarray): 4x2 array of corners in [TL, TR, BR, BL] order.

    Returns:
        np.ndarray: The transformed (warped) image.
    """
    warper = PerspectiveWarper()
    result = warper.warp_adaptive(image, pts)
    if result is None:
        raise ValueError("Transformation failed.")
    return result


def compute_warp_quality(original_corners: np.ndarray, target_corners: np.ndarray) -> float:
    """
    Compute a quality score (0-1) for the perspective warp based on corner mapping.

    Args:
        original_corners (np.ndarray): Original 4 corners.
        target_corners (np.ndarray): Target 4 corners.

    Returns:
        float: Quality score between 0.0 and 1.0.
    """
    try:
        # A simple metric: compute the homography and check reprojection error
        matrix, _ = cv2.findHomography(original_corners, target_corners)
        if matrix is None:
            return 0.0
            
        # Transform original corners
        pts = original_corners.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, matrix).reshape(4, 2)
        
        # Compute mean squared error
        mse = np.mean(np.linalg.norm(target_corners - transformed, axis=1))
        
        # Convert MSE to a 0-1 score (heuristic)
        score = math.exp(-mse / 5.0)
        return float(np.clip(score, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Failed to compute warp quality: {e}")
        return 0.0
