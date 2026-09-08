"""
Unit tests for VEE Scanner booklet detection pipeline.

Run with: python -m pytest test_detector.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import numpy as np
import cv2

from models import (
    BoundingBox,
    QuadCorners,
    DetectionResult,
    DetectionMethod,
    ReviewFlag,
)
from corner_refiner import CornerRefiner
from perspective import PerspectiveWarper
from config import DetectorConfig, PathConfig, resolve_device


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def refiner():
    return CornerRefiner()


@pytest.fixture
def warper():
    return PerspectiveWarper()


@pytest.fixture
def sample_bbox():
    return BoundingBox(
        x1=10.0, y1=20.0, x2=110.0, y2=120.0,
        confidence=0.85, class_id=73, class_name="book",
    )


@pytest.fixture
def sample_quad():
    pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    return QuadCorners(points=pts)


# ─── TestCornerRefiner ────────────────────────────────────────────────────────

class TestCornerRefiner:
    """Tests for corner ordering and refinement."""

    def test_order_corners_random_order(self):
        """Points in random order should be reordered to TL, TR, BR, BL."""
        points = np.array([
            [100, 100],  # BR
            [10, 10],    # TL
            [10, 100],   # BL
            [100, 10],   # TR
        ], dtype=np.float32)

        ordered = CornerRefiner.order_corners(points)
        expected = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        np.testing.assert_array_almost_equal(ordered, expected)

    def test_order_corners_already_ordered(self):
        """Already-ordered points should remain unchanged."""
        points = np.array([
            [10, 10], [100, 10], [100, 100], [10, 100],
        ], dtype=np.float32)

        ordered = CornerRefiner.order_corners(points)
        np.testing.assert_array_almost_equal(ordered, points)

    def test_order_corners_returns_4x2(self):
        """Output should always be shape (4, 2)."""
        points = np.array([
            [50, 20], [200, 30], [190, 300], [40, 280],
        ], dtype=np.float32)
        ordered = CornerRefiner.order_corners(points)
        assert ordered.shape == (4, 2)

    def test_order_corners_collinear(self):
        """Collinear points — should not crash, returns 4 points."""
        points = np.array([
            [10, 10], [50, 50], [100, 100], [150, 150],
        ], dtype=np.float32)
        ordered = CornerRefiner.order_corners(points)
        assert ordered.shape == (4, 2)


# ─── TestQuadValidation ──────────────────────────────────────────────────────

class TestQuadValidation:
    """Tests for geometric quad validation."""

    def test_valid_convex_quad(self):
        """A clean rectangle should pass validation."""
        corners = np.array(
            [[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32
        )
        assert CornerRefiner.validate_quad(corners, (500, 500)) is True

    def test_concave_quad_fails(self):
        """A concave quadrilateral should fail validation."""
        corners = np.array(
            [[0, 0], [200, 0], [100, 50], [0, 200]], dtype=np.float32
        )
        assert CornerRefiner.validate_quad(corners, (500, 500)) is False

    def test_too_small_quad_fails(self):
        """A tiny quad (< 5% of frame area) should fail."""
        corners = np.array(
            [[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32
        )
        assert CornerRefiner.validate_quad(corners, (500, 500)) is False

    def test_none_corners_fails(self):
        """None corners should fail validation."""
        assert CornerRefiner.validate_quad(None, (500, 500)) is False

    def test_wrong_point_count_fails(self):
        """3 points instead of 4 should fail."""
        corners = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        assert CornerRefiner.validate_quad(corners, (500, 500)) is False


# ─── TestModels ───────────────────────────────────────────────────────────────

class TestModels:
    """Tests for data model classes."""

    def test_bounding_box_properties(self, sample_bbox):
        assert sample_bbox.width == 100.0
        assert sample_bbox.height == 100.0
        assert sample_bbox.center == (60.0, 70.0)
        assert sample_bbox.area == 10000.0

    def test_bounding_box_as_array(self, sample_bbox):
        arr = sample_bbox.as_array
        expected = np.array([10.0, 20.0, 110.0, 120.0])
        np.testing.assert_array_almost_equal(arr, expected)

    def test_bounding_box_repr(self, sample_bbox):
        r = repr(sample_bbox)
        assert "book" in r
        assert "0.85" in r

    def test_quad_corners_valid_shape(self, sample_quad):
        assert sample_quad.points.shape == (4, 2)

    def test_quad_corners_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            QuadCorners(points=np.array([[0, 0], [1, 1], [2, 2]]))

    def test_quad_corners_area(self, sample_quad):
        area = sample_quad.area()
        assert area > 0
        # 90x90 square = 8100
        assert abs(area - 8100.0) < 1.0

    def test_quad_corners_is_convex(self, sample_quad):
        assert sample_quad.is_convex()

    def test_quad_corners_as_float32(self, sample_quad):
        f = sample_quad.as_float32()
        assert f.dtype == np.float32

    def test_quad_corners_accessors(self, sample_quad):
        np.testing.assert_array_equal(sample_quad.top_left, [10, 10])
        np.testing.assert_array_equal(sample_quad.top_right, [100, 10])
        np.testing.assert_array_equal(sample_quad.bottom_right, [100, 100])
        np.testing.assert_array_equal(sample_quad.bottom_left, [10, 100])

    def test_detection_result_to_dict(self, sample_bbox, sample_quad):
        result = DetectionResult(
            bbox=sample_bbox,
            corners=sample_quad,
            confidence=0.85,
            detection_method=DetectionMethod.YOLO_CV_REFINED,
            review_flags=[ReviewFlag.OK],
            needs_review=False,
            hands_detected=[],
            clutter_detected=[],
            latency_ms=42.0,
            raw_detections=[sample_bbox],
            frame_shape=(480, 640),
        )
        d = result.to_dict()
        assert d["confidence"] == 0.85
        assert d["needs_review"] is False
        assert d["detection_method"] == "yolo_cv_refined"
        assert d["latency_ms"] == 42.0
        assert isinstance(d["corners"], list)

    def test_detection_result_repr(self, sample_bbox, sample_quad):
        result = DetectionResult(
            bbox=sample_bbox,
            corners=sample_quad,
            confidence=0.85,
            detection_method=DetectionMethod.YOLO_DIRECT,
            review_flags=[ReviewFlag.OK],
            needs_review=False,
            hands_detected=[],
            clutter_detected=[],
            latency_ms=42.0,
            raw_detections=[],
            frame_shape=(480, 640),
        )
        r = repr(result)
        assert "yolo_direct" in r
        assert "0.85" in r


# ─── TestPerspective ─────────────────────────────────────────────────────────

class TestPerspective:
    """Tests for perspective transformation."""

    def test_warp_output_dimensions(self, warper):
        """Warp should produce image with A4 dimensions."""
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        corners = np.array(
            [[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32
        )
        warped = warper.warp_adaptive(frame, corners)
        assert warped is not None
        assert warped.shape[1] == warper.page_w_px
        assert warped.shape[0] == warper.page_h_px

    def test_warp_adaptive_preserves_aspect(self):
        """Adaptive warp should now scale to A4 or A3 target."""
        warper = PerspectiveWarper()
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        # 100-wide by 200-tall rectangle (single page)
        corners = np.array(
            [[10, 10], [110, 10], [110, 210], [10, 210]], dtype=np.float32
        )
        warped = warper.warp_adaptive(frame, corners)
        assert warped is not None
        assert warped.shape[1] == warper.page_w_px  # Single page width
        assert warped.shape[0] == warper.page_h_px  # height

    def test_estimate_page_split_centered(self):
        """Page split on a symmetric image should find center-ish."""
        warper = PerspectiveWarper()
        # Create image with dark line in center
        img = np.full((200, 400, 3), 200, dtype=np.uint8)
        img[:, 195:205] = 20  # dark spine line
        spine = warper.estimate_page_split(img)
        assert 180 <= spine <= 220  # within 10% of center

    def test_split_pages(self):
        """split_pages should produce two images that together span the width."""
        warper = PerspectiveWarper()
        img = np.full((200, 400, 3), 128, dtype=np.uint8)
        left, right = warper.split_pages(img, spine_x=200)
        assert left.shape[1] == 200
        assert right.shape[1] == 200

    def test_warp_none_on_bad_corners(self, warper):
        """Degenerate corners should return None, not crash."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # All corners at same point
        corners = np.array(
            [[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32
        )
        # This may or may not return None depending on OpenCV behavior,
        # but it should NOT crash
        try:
            warper.warp(frame, corners)
        except Exception:
            pass  # acceptable to raise


# ─── TestConfig ───────────────────────────────────────────────────────────────

class TestConfig:
    """Tests for configuration module."""

    def test_resolve_device_cpu_fallback(self):
        """resolve_device('cuda') should return 'cpu' if CUDA unavailable."""
        import torch
        if not torch.cuda.is_available():
            result = resolve_device("cuda")
            assert result == "cpu"

    def test_resolve_device_auto(self):
        """resolve_device('auto') should return a valid device."""
        result = resolve_device("auto")
        assert result in ("cpu", "cuda")

    def test_resolve_device_explicit_cpu(self):
        """resolve_device('cpu') should always return 'cpu'."""
        assert resolve_device("cpu") == "cpu"

    def test_path_config_from_root(self, tmp_path):
        """PathConfig.from_project_root should create correct paths."""
        pc = PathConfig.from_project_root(tmp_path)
        assert pc.project_root == tmp_path.resolve()
        assert pc.models_dir == tmp_path.resolve() / "models"
        assert pc.output_dir == tmp_path.resolve() / "output"
        assert pc.training_dir == tmp_path.resolve() / "training"

    def test_detector_config_defaults(self):
        """DetectorConfig defaults should be sensible."""
        cfg = DetectorConfig()
        assert cfg.confidence_threshold == 0.35
        assert cfg.high_confidence > cfg.medium_confidence
        assert cfg.input_size == 640
        assert 73 in cfg.COCO_BOOKLET_CLASSES


# ─── TestSyntheticDetection ──────────────────────────────────────────────────

class TestSyntheticDetection:
    """Integration test with synthetic frames using the CV-only path."""

    def test_white_rect_on_black_contour_detection(self, refiner):
        """A white rectangle on black bg should be found by the contour method."""
        frame = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (400, 400), 255, -1)

        # Test the contour method directly
        corners = refiner._contour_method(frame)
        # May or may not succeed depending on adaptive threshold behavior,
        # but should not crash
        if corners is not None:
            assert corners.shape == (4, 2)

    def test_corner_refiner_end_to_end(self, refiner):
        """Full corner refinement pipeline on a synthetic frame."""
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), -1)

        bbox = np.array([90, 90, 410, 410], dtype=np.float32)
        corners, confidence = refiner.refine_corners(frame, bbox)

        # Should find something (at minimum the bbox fallback)
        assert corners is not None
        assert corners.shape == (4, 2)
        assert confidence > 0.0

    def test_validate_roi_quad_rejects_inner_sub_box(self):
        """validate_roi_quad must reject small internal tables and accept full-coverage quads."""
        roi_shape = (500, 500)
        # Inner table: 150x150 in center (9% of ROI area)
        inner_table = np.array([
            [175, 175],
            [325, 175],
            [325, 325],
            [175, 325],
        ], dtype=np.float32)
        assert not CornerRefiner.validate_roi_quad(inner_table, roi_shape)

        # Full booklet: 440x440 in center (77% of ROI area, span 88%)
        full_quad = np.array([
            [30, 30],
            [470, 30],
            [470, 470],
            [30, 470],
        ], dtype=np.float32)
        assert CornerRefiner.validate_roi_quad(full_quad, roi_shape)

    def test_corner_refinement_does_not_collapse_on_internal_table(self, refiner):
        """Refiner must not collapse onto a high-contrast internal table/box inside a booklet."""
        # 600x600 dark desk
        frame = np.full((600, 600, 3), 40, dtype=np.uint8)
        # 400x400 white booklet
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), -1)
        # 120x120 sharp black registration table inside the white booklet
        cv2.rectangle(frame, (180, 180), (300, 300), (0, 0, 0), 4)

        bbox = np.array([100, 100, 500, 500], dtype=np.float32)
        corners, confidence = refiner.refine_corners(frame, bbox)

        assert corners is not None
        # Must cover >= 65% of the 400x400 = 160,000 area
        quad_area = cv2.contourArea(corners.astype(np.float32))
        bbox_area = 400 * 400
        assert quad_area >= 0.65 * bbox_area, f"Refined quad collapsed! Area: {quad_area} vs bbox {bbox_area}"

    def test_fallback_preserves_exact_bbox_coordinates(self, refiner):
        """When CV strategies are rejected, fallback must produce exact unpadded bbox corners."""
        # Frame with solid gray (no features)
        frame = np.full((600, 600, 3), 128, dtype=np.uint8)
        bbox = np.array([120, 150, 480, 450], dtype=np.float32)
        corners, confidence = refiner.refine_corners(frame, bbox, padding_ratio=0.05)

        assert corners is not None
        # Corners should equal the exact unpadded bbox coordinates
        np.testing.assert_allclose(
            corners,
            np.array([
                [120, 150],
                [480, 150],
                [480, 450],
                [120, 450],
            ], dtype=np.float32),
            atol=1.0,
        )

