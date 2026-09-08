"""
BookletDetector: YOLO11-based booklet boundary detection for VEE Scanner.

This is the main detection module. It combines YOLO11 object detection (for
robust coarse localization) with classical CV corner refinement (for precise
quad estimation), plus confidence-gated fallback logic.

Phase 1 (immediate): Uses COCO-pretrained YOLO11 to detect 'book' class (73).
Phase 2 (after training): Uses custom fine-tuned model for booklet/hand/clutter.

Usage:
    from detector import BookletDetector
    detector = BookletDetector()
    result = detector.detect(frame)
    if result.corners is not None and not result.needs_review:
        warped = detector.warp(frame, result)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import DetectorConfig, resolve_device
from corner_refiner import CornerRefiner
from models import (
    BoundingBox,
    DetectionMethod,
    DetectionResult,
    QuadCorners,
    ReviewFlag,
)
from perspective import PerspectiveWarper

logger = logging.getLogger(__name__)


class BookletDetector:
    """YOLO11-based booklet boundary detector with CV corner refinement.

    Architecture:
        1. YOLO11 inference → coarse bounding box for booklet
        2. Corner refinement → precise 4-corner quad within the YOLO bbox
        3. Confidence gate → high/medium/low confidence handling
        4. Perspective warp → flatten booklet to rectangular image

    The key insight: contour detection inside a YOLO-detected ROI is far more
    reliable than on the full frame, because desk texture, distant shadows,
    and clutter outside the ROI are eliminated.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """Initialize the BookletDetector.

        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self.config = config or DetectorConfig()
        self._device = resolve_device(self.config.device)
        self._model = None
        self._is_custom_model = False
        self._corner_refiner = CornerRefiner()
        self._warper = PerspectiveWarper()

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO11 model (pretrained or custom).

        Tries custom model first if configured, falls back to pretrained COCO.
        Supports .pt, .onnx, and .engine model formats.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics>=8.3.0"
            )

        # Try custom model first (Phase 2)
        if self.config.custom_model_path and Path(self.config.custom_model_path).exists():
            model_path = self.config.custom_model_path
            self._is_custom_model = True
            logger.info(f"Loading custom model: {model_path}")
        else:
            model_path = self.config.model_path
            self._is_custom_model = False
            logger.info(f"Loading pretrained model: {model_path}")

        self._model = YOLO(model_path)
        logger.info(
            f"Model loaded on device '{self._device}' "
            f"(custom={self._is_custom_model})"
        )

    # ------------------------------------------------------------------
    # Main detection pipeline
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run the full booklet detection pipeline on a single frame.

        Pipeline:
            1. Run YOLO11 inference
            2. Filter detections for booklet class
            3. Detect hands/clutter (if present)
            4. Apply confidence-gated corner refinement
            5. Return structured DetectionResult

        Args:
            frame: Input image (BGR, any resolution).

        Returns:
            DetectionResult with bbox, corners, confidence, and flags.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        review_flags: list[ReviewFlag] = []

        # ── Step 1: YOLO inference ────────────────────────────────────
        raw_detections = self._run_yolo(frame)
        hands = [d for d in raw_detections if self._is_hand(d)]
        clutter = [d for d in raw_detections if self._is_clutter(d)]
        booklet_detections = [d for d in raw_detections if self._is_booklet(d)]

        # ── Step 2: Select best booklet detection ─────────────────────
        if not booklet_detections:
            logger.info("No booklet detected by YOLO — attempting CV fallback.")
            return self._cv_only_fallback(
                frame, hands, clutter, raw_detections, t0
            )

        # Pick highest-confidence booklet
        booklet_detections.sort(key=lambda d: d.confidence, reverse=True)
        best = booklet_detections[0]

        if len(booklet_detections) > 1:
            review_flags.append(ReviewFlag.MULTIPLE_BOOKLETS)

        # ── Step 3: Hand occlusion check ──────────────────────────────
        if hands:
            review_flags.append(ReviewFlag.HAND_OCCLUSION)
            logger.info(f"Hand(s) detected ({len(hands)}). Will mask during refinement.")

        # ── Step 4: Confidence-gated corner refinement ────────────────
        def _covers_bbox(corners_pts: np.ndarray, bbox: BoundingBox) -> bool:
            if corners_pts is None or len(corners_pts) != 4:
                return False
            quad_area = QuadCorners(corners_pts).area()
            bbox_area = bbox.area
            if bbox_area <= 0:
                return False
            area_ratio = quad_area / bbox_area
            # Quad area must be within 65% to 135% of the YOLO bbox area
            if area_ratio < 0.65 or area_ratio > 1.35:
                return False
            # Quad span must cover at least 75% of YOLO bbox width and height
            xs = corners_pts[:, 0]
            ys = corners_pts[:, 1]
            span_w = float(xs.max() - xs.min())
            span_h = float(ys.max() - ys.min())
            if span_w < 0.75 * bbox.width or span_h < 0.75 * bbox.height:
                return False
            return True

        corners, refine_conf = self._corner_refiner.refine_corners(
            frame, best.as_array, self.config.roi_padding
        )

        if corners is not None and _covers_bbox(corners, best) and refine_conf > 0.5:
            method = DetectionMethod.YOLO_CV_REFINED
        else:
            if corners is not None and not _covers_bbox(corners, best):
                logger.warning(
                    f"CV refined quad failed bbox coverage check (ratio={QuadCorners(corners).area()/best.area:.2f}). "
                    "Falling back to YOLO bounding box to ensure full booklet capture."
                )
            corners = self._bbox_to_corners(best)
            method = DetectionMethod.YOLO_DIRECT

        if best.confidence < self.config.medium_confidence:
            review_flags.append(ReviewFlag.LOW_CONFIDENCE)

        # ── Step 5: Geometric validation ──────────────────────────────
        if corners is not None:
            if not self._corner_refiner.validate_quad(corners, (h, w)):
                review_flags.append(ReviewFlag.GEOMETRIC_INVALID)
                logger.warning("Detected quad failed geometric validation.")
                # Still return the corners — let caller decide
        else:
            review_flags.append(ReviewFlag.NO_DETECTION)

        # ── Build result ──────────────────────────────────────────────
        needs_review = any(
            f in review_flags
            for f in (
                ReviewFlag.LOW_CONFIDENCE,
                ReviewFlag.GEOMETRIC_INVALID,
                ReviewFlag.NO_DETECTION,
            )
        )
        latency = (time.perf_counter() - t0) * 1000.0

        quad = QuadCorners(corners) if corners is not None else None

        return DetectionResult(
            bbox=best,
            corners=quad,
            confidence=best.confidence,
            detection_method=method,
            review_flags=review_flags if review_flags else [ReviewFlag.OK],
            needs_review=needs_review,
            hands_detected=hands,
            clutter_detected=clutter,
            latency_ms=latency,
            raw_detections=raw_detections,
            frame_shape=(h, w),
        )

    # ------------------------------------------------------------------
    # YOLO inference
    # ------------------------------------------------------------------

    def _run_yolo(self, frame: np.ndarray) -> list[BoundingBox]:
        """Run YOLO11 inference and return parsed detections.

        Args:
            frame: Input image (BGR).

        Returns:
            List of BoundingBox objects for all detected objects.
        """
        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            device=self._device,
            verbose=False,
        )

        detections: list[BoundingBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                box = result.boxes[i]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, str(cls_id))

                detections.append(
                    BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Class filtering helpers
    # ------------------------------------------------------------------

    def _is_booklet(self, det: BoundingBox) -> bool:
        """Check if a detection is a booklet (COCO 'book' or custom 'booklet')."""
        if self._is_custom_model:
            return det.class_name == "booklet"
        return det.class_id in self.config.COCO_BOOKLET_CLASSES

    def _is_hand(self, det: BoundingBox) -> bool:
        """Check if a detection is a hand (COCO 'person' or custom 'hand')."""
        if self._is_custom_model:
            return det.class_name == "hand"
        # In COCO, there's no 'hand' class — 'person' is class 0
        # We use a heuristic: small person detections near the booklet are likely hands
        return det.class_id == self.config.COCO_HAND_CLASS

    def _is_clutter(self, det: BoundingBox) -> bool:
        """Check if a detection is background clutter."""
        if self._is_custom_model:
            return det.class_name == "background_clutter"
        # In COCO pretrained mode, clutter items might be: cell phone (67),
        # scissors (76), etc. — not critical for Phase 1.
        return False

    # ------------------------------------------------------------------
    # Fallback strategies
    # ------------------------------------------------------------------

    def _cv_only_fallback(
        self,
        frame: np.ndarray,
        hands: list[BoundingBox],
        clutter: list[BoundingBox],
        raw_detections: list[BoundingBox],
        t0: float,
    ) -> DetectionResult:
        """Full-frame CV fallback when YOLO detects no booklet.

        Uses adaptive thresholding + contour on the entire frame as a last resort.
        Always flags for operator review.

        Args:
            frame: Input image (BGR).
            hands: Detected hand bounding boxes.
            clutter: Detected clutter bounding boxes.
            raw_detections: All raw YOLO detections.
            t0: Pipeline start time (perf_counter).

        Returns:
            DetectionResult from CV fallback.
        """
        h, w = frame.shape[:2]

        # Try contour on full frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        corners = None
        bbox = None
        method = DetectionMethod.NONE

        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            frame_area = h * w

            if area > frame_area * self.config.min_booklet_area_ratio:
                epsilon = 0.02 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)

                if len(approx) == 4:
                    corners = CornerRefiner.order_corners(
                        approx.reshape(4, 2).astype(np.float32)
                    )
                    method = DetectionMethod.CV_FALLBACK

                    # Construct a bounding box from the contour
                    x, y, bw, bh = cv2.boundingRect(largest)
                    bbox = BoundingBox(
                        x1=float(x), y1=float(y),
                        x2=float(x + bw), y2=float(y + bh),
                        confidence=0.0, class_id=-1, class_name="cv_fallback",
                    )

        latency = (time.perf_counter() - t0) * 1000.0
        quad = QuadCorners(corners) if corners is not None else None

        review_flags = [ReviewFlag.NO_DETECTION]
        if method == DetectionMethod.CV_FALLBACK:
            review_flags = [ReviewFlag.LOW_CONFIDENCE]

        return DetectionResult(
            bbox=bbox,
            corners=quad,
            confidence=0.0,
            detection_method=method,
            review_flags=review_flags,
            needs_review=True,  # always flag CV-only results for review
            hands_detected=hands,
            clutter_detected=clutter,
            latency_ms=latency,
            raw_detections=raw_detections,
            frame_shape=(h, w),
        )

    def _bbox_to_corners(self, bbox: BoundingBox) -> np.ndarray:
        """Convert a bounding box to 4-corner array (TL, TR, BR, BL).

        Args:
            bbox: BoundingBox to convert.

        Returns:
            4x2 numpy array of corners.
        """
        return np.array([
            [bbox.x1, bbox.y1],  # TL
            [bbox.x2, bbox.y1],  # TR
            [bbox.x2, bbox.y2],  # BR
            [bbox.x1, bbox.y2],  # BL
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def warp(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        adaptive: bool = True,
    ) -> Optional[np.ndarray]:
        """Warp the booklet region to a flat rectangular image.

        Args:
            frame: Original input frame.
            result: DetectionResult from detect().
            adaptive: If True, preserve the booklet's natural aspect ratio.

        Returns:
            Warped booklet image, or None if corners are unavailable.
        """
        if result.corners is None:
            return None
        corners = result.corners.as_float32()
        if adaptive:
            return self._warper.warp_adaptive(frame, corners)
        return self._warper.warp(frame, corners)

    def detect_and_warp(
        self, frame: np.ndarray
    ) -> tuple[DetectionResult, Optional[np.ndarray]]:
        """Convenience method: detect booklet and warp in one call.

        Args:
            frame: Input image (BGR).

        Returns:
            Tuple of (DetectionResult, warped_image_or_None).
        """
        result = self.detect(frame)
        warped = self.warp(frame, result)
        return result, warped

    def get_model_info(self) -> dict:
        """Return information about the loaded model.

        Returns:
            Dict with model path, device, custom status, etc.
        """
        return {
            "model_path": (
                self.config.custom_model_path
                if self._is_custom_model
                else self.config.model_path
            ),
            "device": self._device,
            "is_custom_model": self._is_custom_model,
            "input_size": self.config.input_size,
            "confidence_threshold": self.config.confidence_threshold,
        }
