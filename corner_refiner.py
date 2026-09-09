import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Constants
MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 0.99
MIN_ANGLE_DEG = 30.0
MAX_ASPECT_RATIO = 3.0
MIN_ROI_COVERAGE = 0.40  # candidate quad must cover >=40% of the cropped ROI
MIN_ROI_SPAN = 0.55      # candidate quad must span >=55% of ROI width and height

HOUGH_CONFIDENCE = 1.0
CONTOUR_CONFIDENCE = 0.8
CORNER_CONFIDENCE = 0.6
BBOX_CONFIDENCE = 0.5



class CornerRefiner:
    """
    Robust corner refinement for booklet detection.

    Given a coarse bounding box, it finds precise 4-corner coordinates
    using multiple classical CV strategies within the cropped ROI.
    """

    def __init__(self) -> None:
        """Initialize the CornerRefiner."""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def refine_corners(
        self, frame: np.ndarray, bbox_xyxy: np.ndarray, padding_ratio: float = 0.05
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Refine the corners of the booklet within the bounding box.

        Args:
            frame: Full image frame (BGR).
            bbox_xyxy: Bounding box [x1, y1, x2, y2].
            padding_ratio: Ratio to pad the bounding box for the ROI.

        Returns:
            Tuple of (refined_corners, confidence), where refined_corners is a
            4x2 numpy array of (x, y) coordinates, or (None, 0.0) if all strategies fail.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox_xyxy[:4])
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w <= 0 or bbox_h <= 0:
                logger.warning("Invalid bounding box dimensions.")
                return None, 0.0

            roi, offset_x, offset_y = self._crop_roi(frame, bbox_xyxy, padding_ratio)
            if roi.size == 0:
                logger.warning("Empty ROI cropped.")
                return None, 0.0

            roi_gray = self._preprocess_roi(roi)

            # Strategy A: Contour method (Otsu paper segmentation preferred)
            corners = self._contour_method(roi_gray, roi_bgr=roi)
            confidence = CONTOUR_CONFIDENCE

            # Strategy B: Hough line intersection
            if corners is None or not self.validate_roi_quad(corners, roi_gray.shape):
                corners = self._hough_line_method(roi_gray)
                confidence = HOUGH_CONFIDENCE

            # Strategy C: Shi-Tomasi corner detection + convex hull
            if corners is None or not self.validate_roi_quad(corners, roi_gray.shape):
                corners = self._corner_detection_method(roi_gray)
                confidence = CORNER_CONFIDENCE

            # Strategy D: Fallback to exact unpadded bbox corners mapped into ROI
            if corners is None or not self.validate_roi_quad(corners, roi_gray.shape):
                pad_x = max(0, x1 - offset_x)
                pad_y = max(0, y1 - offset_y)
                corners = np.array(
                    [
                        [pad_x, pad_y],
                        [pad_x + bbox_w, pad_y],
                        [pad_x + bbox_w, pad_y + bbox_h],
                        [pad_x, pad_y + bbox_h],
                    ],
                    dtype=np.float32,
                )
                confidence = BBOX_CONFIDENCE

            corners = self.order_corners(corners)
            full_frame_corners = self._map_corners_to_frame(corners, offset_x, offset_y)

            # Final validation on full frame
            if self.validate_quad(full_frame_corners, frame.shape[:2]):
                return full_frame_corners, confidence
            else:
                logger.warning("Final corners failed validation.")
                return None, 0.0

        except Exception as e:
            logger.error(f"Error during corner refinement: {e}")
            return None, 0.0

    def _crop_roi(
        self, frame: np.ndarray, bbox: np.ndarray, padding_ratio: float
    ) -> Tuple[np.ndarray, int, int]:
        """Crop the Region of Interest from the frame with padding."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]

        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)

        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        return roi, x1_pad, y1_pad

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Convert ROI to grayscale and enhance contrast."""
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
        
        roi_enhanced = self.clahe.apply(roi_gray)
        return roi_enhanced

    def _map_corners_to_frame(
        self, corners: np.ndarray, offset_x: int, offset_y: int
    ) -> np.ndarray:
        """Map ROI corners back to full frame coordinates."""
        mapped = corners.copy()
        mapped[:, 0] += offset_x
        mapped[:, 1] += offset_y
        return mapped

    def _hough_line_method(self, roi_gray: np.ndarray) -> Optional[np.ndarray]:
        """Find corners using Hough line intersections."""
        try:
            # Adaptive thresholding and edges
            blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
            )
            if lines is None:
                return None

            groups = self._cluster_lines_by_orientation(lines)
            if not groups["horizontal"] or not groups["vertical"]:
                return None

            intersections = self._find_line_intersections(groups)
            if len(intersections) < 4:
                return None
            
            pts = np.array(intersections, dtype=np.float32)
            
            # If we have more than 4 intersections, take the convex hull and approximate
            if len(pts) > 4:
                hull = cv2.convexHull(pts)
                epsilon = 0.05 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) == 4:
                    res = approx.reshape(4, 2).astype(np.float32)
                    if self.validate_roi_quad(res, roi_gray.shape):
                        return res
                return None
            elif len(pts) == 4:
                if self.validate_roi_quad(pts, roi_gray.shape):
                    return pts

        except Exception as e:
            logger.debug(f"Hough method failed: {e}")
        return None

    def _cluster_lines_by_orientation(self, lines: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Cluster lines into horizontal and vertical groups."""
        groups = {"horizontal": [], "vertical": []}
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                angle = 90.0
            else:
                angle = math.degrees(math.atan(abs(y2 - y1) / abs(x2 - x1)))

            if angle < 45:
                groups["horizontal"].append(line[0])
            else:
                groups["vertical"].append(line[0])
        return groups

    def _find_line_intersections(self, groups: Dict[str, List[np.ndarray]]) -> List[Tuple[float, float]]:
        """Find intersections between horizontal and vertical lines."""
        intersections = []
        # Simple approach: average horizontal lines and vertical lines? 
        # Better: find extreme lines (top, bottom, left, right)
        
        horiz = groups["horizontal"]
        vert = groups["vertical"]
        
        if not horiz or not vert:
            return intersections

        # Sort horizontal lines by y-coordinate (midpoint) to find top and bottom
        horiz.sort(key=lambda l: (l[1] + l[3]) / 2)
        top_lines = [l for l in horiz if (l[1] + l[3]) / 2 < np.mean([(hl[1]+hl[3])/2 for hl in horiz])]
        bottom_lines = [l for l in horiz if l not in top_lines]

        # Sort vertical lines by x-coordinate (midpoint) to find left and right
        vert.sort(key=lambda l: (l[0] + l[2]) / 2)
        left_lines = [l for l in vert if (l[0] + l[2]) / 2 < np.mean([(vl[0]+vl[2])/2 for vl in vert])]
        right_lines = [l for l in vert if l not in left_lines]

        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None
            
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return (px, py)

        def avg_line(lines):
            if not lines:
                return None
            return np.mean(lines, axis=0)

        top = avg_line(top_lines)
        bottom = avg_line(bottom_lines)
        left = avg_line(left_lines)
        right = avg_line(right_lines)

        if top is not None and left is not None:
            pt = line_intersection(top, left)
            if pt: intersections.append(pt)
        if top is not None and right is not None:
            pt = line_intersection(top, right)
            if pt: intersections.append(pt)
        if bottom is not None and right is not None:
            pt = line_intersection(bottom, right)
            if pt: intersections.append(pt)
        if bottom is not None and left is not None:
            pt = line_intersection(bottom, left)
            if pt: intersections.append(pt)

        return intersections

    def _contour_method(
        self, roi_gray: np.ndarray, roi_bgr: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Find corners using largest contour (paper color segmentation + Otsu, adaptive as fallback)."""
        try:
            # If the ROI is flat/featureless (e.g. uniform color or gray), no contour strategy can be trusted
            if float(np.std(roi_gray)) < 5.0 or int(np.ptp(roi_gray)) < 15:
                return None

            blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            # Strategy 1: Paper color segmentation + Otsu binarization + morphological close
            _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if roi_bgr is not None and len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
                hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                # White/off-white paper: low saturation (S < 85), moderate-to-high brightness (V > 70)
                paper_mask = ((s < 85) & (v > 70)).astype(np.uint8) * 255
                seg_mask = thresh_otsu & paper_mask
            else:
                seg_mask = thresh_otsu

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                roi_area = float(roi_gray.shape[0] * roi_gray.shape[1])
                c_top = contours[0]

                # Two-page spread fusion: spine valley often divides paper into 2 large page contours
                if (
                    len(contours) >= 2
                    and cv2.contourArea(contours[1]) > 0.20 * roi_area
                    and cv2.contourArea(c_top) < 0.70 * roi_area
                ):
                    combined_pts = np.vstack([contours[0], contours[1]])
                    hull = cv2.convexHull(combined_pts)
                else:
                    hull = cv2.convexHull(c_top)

                peri = cv2.arcLength(hull, True)
                # Multi-epsilon search: textured/patterned backgrounds or paper wrinkles
                # may produce extra vertices at a single fixed epsilon
                for eps in (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05):
                    approx = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(approx) == 4:
                        pts = approx.reshape(4, 2).astype(np.float32)
                        if self.validate_roi_quad(pts, roi_gray.shape):
                            return pts

                # Fallback: minAreaRect fits the minimum enclosing oriented rectangle to the paper contour
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect).astype(np.float32)
                if self.validate_roi_quad(box, roi_gray.shape):
                    return box

            # Strategy 2: Adaptive thresholding (for low-contrast or textured backgrounds)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                roi_area = float(roi_gray.shape[0] * roi_gray.shape[1])
                c_top = contours[0]
                if (
                    len(contours) >= 2
                    and cv2.contourArea(contours[1]) > 0.20 * roi_area
                    and cv2.contourArea(c_top) < 0.70 * roi_area
                ):
                    combined_pts = np.vstack([contours[0], contours[1]])
                    hull = cv2.convexHull(combined_pts)
                else:
                    hull = cv2.convexHull(c_top)

                peri = cv2.arcLength(hull, True)
                for eps in (0.015, 0.02, 0.025, 0.03, 0.035, 0.04):
                    approx = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(approx) == 4:
                        pts = approx.reshape(4, 2).astype(np.float32)
                        if self.validate_roi_quad(pts, roi_gray.shape):
                            return pts

        except Exception as e:
            logger.debug(f"Contour method failed: {e}")
        return None

    def _corner_detection_method(self, roi_gray: np.ndarray) -> Optional[np.ndarray]:
        """Find corners using Shi-Tomasi and convex hull."""
        try:
            corners = cv2.goodFeaturesToTrack(roi_gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
            if corners is None or len(corners) < 4:
                return None

            corners = corners.reshape(-1, 2)
            hull = cv2.convexHull(corners.astype(np.float32))
            
            epsilon = 0.05 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                
                # Sub-pixel refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.cornerSubPix(roi_gray, pts, (5, 5), (-1, -1), criteria)
                if self.validate_roi_quad(pts, roi_gray.shape):
                    return pts

        except Exception as e:
            logger.debug(f"Corner detection method failed: {e}")
        return None

    @staticmethod
    def order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Order points in Top-Left, Top-Right, Bottom-Right, Bottom-Left order.

        Args:
            pts: Array of 4 points.

        Returns:
            Ordered array of 4 points.
        """
        if len(pts) != 4:
            # Fallback if somehow not 4 points, just return first 4 or pad
            pts = pts[:4] if len(pts) > 4 else np.pad(pts, ((0, 4 - len(pts)), (0, 0)), mode='edge')

        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-Left
        rect[2] = pts[np.argmax(s)]  # Bottom-Right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-Right
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left

        return rect

    @staticmethod
    def validate_quad(
        corners: np.ndarray,
        frame_shape: Tuple[int, int],
        is_roi: bool = False,
    ) -> bool:
        """
        Validate if the 4 points form a valid quadrilateral representing a booklet.

        Args:
            corners: Array of 4 points.
            frame_shape: (height, width) of the frame or ROI.
            is_roi: If True, relaxed upper area bound allows booklet to fill up to 105% of ROI.

        Returns:
            True if valid, False otherwise.
        """
        if corners is None or len(corners) != 4:
            return False

        h, w = frame_shape[:2]
        frame_area = h * w

        # Check bounds
        for x, y in corners:
            if not (-w * 0.1 <= x <= w * 1.1 and -h * 0.1 <= y <= h * 1.1):
                return False

        # Calculate area
        pts = corners.astype(np.float32)
        area = cv2.contourArea(pts)
        min_ratio = 0.20 if is_roi else MIN_AREA_RATIO
        max_ratio = 1.05 if is_roi else MAX_AREA_RATIO
        if area < min_ratio * frame_area or area > max_ratio * frame_area:
            return False

        # Check convexity
        if not cv2.isContourConvex(pts):
            return False

        # Check angles and aspect ratio
        def angle_between(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cosine_angle))

        for i in range(4):
            ang = angle_between(pts[i - 1], pts[i], pts[(i + 1) % 4])
            if ang < MIN_ANGLE_DEG or ang > (180.0 - MIN_ANGLE_DEG):
                return False

        # Check aspect ratio
        w1 = np.linalg.norm(pts[0] - pts[1])
        w2 = np.linalg.norm(pts[2] - pts[3])
        h1 = np.linalg.norm(pts[1] - pts[2])
        h2 = np.linalg.norm(pts[3] - pts[0])
        
        width = max(w1, w2)
        height = max(h1, h2)
        
        if height == 0 or width == 0:
            return False
            
        ar = max(width/height, height/width)
        if ar > MAX_ASPECT_RATIO:
            return False

        return True

    @classmethod
    def validate_roi_quad(
        cls,
        corners: np.ndarray,
        roi_shape: Tuple[int, int],
        min_coverage: float = MIN_ROI_COVERAGE,
        min_span: float = MIN_ROI_SPAN,
    ) -> bool:
        """
        Validate that candidate corners within a cropped ROI represent the full booklet,
        not an internal printed table, text block, or sub-box.

        Args:
            corners: 4x2 array of candidate corners.
            roi_shape: (height, width) of the cropped ROI.
            min_coverage: Minimum ratio of quad area to ROI area (default 0.40).
            min_span: Minimum ratio of quad width/height span to ROI dimensions (default 0.55).

        Returns:
            True if quad spans the full booklet ROI, False otherwise.
        """
        if corners is None or len(corners) != 4:
            return False

        if not cls.validate_quad(corners, roi_shape, is_roi=True):
            return False

        h, w = roi_shape[:2]
        roi_area = h * w
        if roi_area <= 0:
            return False

        pts = corners.astype(np.float32)
        area = cv2.contourArea(pts)
        if area < min_coverage * roi_area:
            return False

        xs = pts[:, 0]
        ys = pts[:, 1]
        span_w = float(xs.max() - xs.min())
        span_h = float(ys.max() - ys.min())
        if span_w < min_span * w or span_h < min_span * h:
            return False

        return True

