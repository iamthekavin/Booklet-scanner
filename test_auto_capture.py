"""
Unit tests for AutoCaptureController.

Run with: python -m pytest test_auto_capture.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import numpy as np

from auto_capture import AutoCaptureController, AutoCaptureConfig, AutoCaptureState
from models import DetectionResult, DetectionMethod, ReviewFlag, QuadCorners, BoundingBox


def make_result(
    corners_pts: np.ndarray,
    conf: float = 0.85,
    hands: bool = False,
) -> DetectionResult:
    """Helper to construct a mock DetectionResult."""
    quad = QuadCorners(points=corners_pts.astype(np.float32))
    bbox = BoundingBox(
        x1=float(corners_pts[:, 0].min()),
        y1=float(corners_pts[:, 1].min()),
        x2=float(corners_pts[:, 0].max()),
        y2=float(corners_pts[:, 1].max()),
        confidence=conf,
        class_id=0,
        class_name="booklet",
    )
    hands_list = [bbox] if hands else []
    flags = [ReviewFlag.HAND_OCCLUSION] if hands else [ReviewFlag.OK]

    return DetectionResult(
        bbox=bbox,
        corners=quad,
        confidence=conf,
        detection_method=DetectionMethod.YOLO_CV_REFINED,
        review_flags=flags,
        needs_review=hands,
        hands_detected=hands_list,
        clutter_detected=[],
        latency_ms=10.0,
        raw_detections=[bbox],
        frame_shape=(720, 1280),
    )


class TestAutoCaptureController:
    """Tests for auto-capture stability tracking and state machine."""

    @pytest.fixture
    def controller(self):
        cfg = AutoCaptureConfig(
            stability_duration=0.5,      # Fast 0.5s for testing
            drift_threshold_px=10.0,
            min_confidence=0.50,
            cooldown_duration=0.5,
            page_turn_drift_px=30.0,
            require_no_hands=True,
        )
        return AutoCaptureController(config=cfg, enabled=True)

    @pytest.fixture
    def base_corners(self):
        return np.array([
            [100.0, 100.0],
            [400.0, 100.0],
            [400.0, 500.0],
            [100.0, 500.0],
        ], dtype=np.float32)

    def test_disabled_toggle(self, controller):
        """Controller toggle on/off correctly disables updates."""
        assert controller.enabled is True
        controller.toggle()
        assert controller.enabled is False
        assert controller.state == AutoCaptureState.DISABLED

        captured, state, prog, msg = controller.update(None)
        assert captured is False
        assert state == AutoCaptureState.DISABLED

    def test_idle_when_no_detection(self, controller):
        """Without valid detection, stays in IDLE."""
        captured, state, prog, msg = controller.update(None, q_pass=True, dt=0.1)
        assert captured is False
        assert state == AutoCaptureState.IDLE

    def test_stabilization_accumulates_timer(self, controller, base_corners):
        """Stationary booklet accumulates stability timer toward 1.0."""
        res = make_result(base_corners)

        # Frame 1: initial registration
        captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.1)
        assert captured is False
        assert state == AutoCaptureState.STABILIZING

        # Frame 2: same corners, dt=0.2s -> 40% of 0.5s
        captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.2)
        assert captured is False
        assert state == AutoCaptureState.STABILIZING
        assert 0.35 <= prog <= 0.45

    def test_large_drift_resets_timer(self, controller, base_corners):
        """Jumping/moving corners resets stability progress."""
        res1 = make_result(base_corners)
        controller.update(res1, q_pass=True, dt=0.3)

        # Move corners by 25px (> 10px drift threshold)
        moved_corners = base_corners + 25.0
        res2 = make_result(moved_corners)

        captured, state, prog, msg = controller.update(res2, q_pass=True, dt=0.1)
        assert captured is False
        assert prog < 0.2  # Timer was heavily reduced or reset

    def test_quality_gate_failure_resets(self, controller, base_corners):
        """Blurry frame resets stabilization timer back to 0."""
        res = make_result(base_corners)
        controller.update(res, q_pass=True, dt=0.3)

        # Blurry frame (q_pass=False)
        captured, state, prog, msg = controller.update(res, q_pass=False, dt=0.1)
        assert captured is False
        assert state == AutoCaptureState.IDLE
        assert controller.stable_timer == 0.0

    def test_hand_occlusion_resets(self, controller, base_corners):
        """Hand presence resets stabilization."""
        res_clean = make_result(base_corners, hands=False)
        controller.update(res_clean, q_pass=True, dt=0.3)

        # Hand enters
        res_hands = make_result(base_corners, hands=True)
        captured, state, prog, msg = controller.update(res_hands, q_pass=True, dt=0.1)
        assert captured is False
        assert state == AutoCaptureState.IDLE
        assert "Hand detected" in msg

    def test_trigger_and_cooldown_cycle(self, controller, base_corners):
        """When steady for stability_duration, trigger fires once then enters cooldown."""
        res = make_result(base_corners)

        # Frame 1: establishes baseline corners
        controller.update(res, q_pass=True, dt=0.1)
        # Frame 2: steady 0.3s
        controller.update(res, q_pass=True, dt=0.3)
        # Frame 3: steady 0.3s -> total 0.6s >= 0.5s stability
        captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.3)
        assert captured is True
        assert state == AutoCaptureState.TRIGGERED
        assert prog == 1.0

        # Frame 4: enters COOLDOWN
        captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.1)
        assert captured is False
        assert state == AutoCaptureState.COOLDOWN

    def test_anti_duplicate_requires_page_turn(self, controller, base_corners):
        """Must not re-capture the same static page after cooldown without a page turn."""
        res = make_result(base_corners)

        # 1. Trigger capture
        controller.update(res, q_pass=True, dt=0.1)
        controller.update(res, q_pass=True, dt=0.3)
        captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.3)
        assert captured is True
        assert state == AutoCaptureState.TRIGGERED

        # 2. Advance through cooldown (cooldown_duration=0.5s)
        controller.update(res, q_pass=True, dt=0.1)   # COOLDOWN
        controller.update(res, q_pass=True, dt=0.5)   # Exit cooldown -> WAITING_FOR_PAGE_TURN

        assert controller.state == AutoCaptureState.WAITING_FOR_PAGE_TURN

        # 3. Same static page remains on desk for 2 more seconds -> NEVER captures again!
        for _ in range(10):
            captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.2)
            assert captured is False
            assert state == AutoCaptureState.WAITING_FOR_PAGE_TURN

        # 4. User turns page (displacement > 30px or booklet temporarily removed)
        turned_corners = base_corners + 50.0
        res_turned = make_result(turned_corners)
        controller.update(res_turned, q_pass=True, dt=0.1)

        # Now re-armed!
        assert controller.state in (AutoCaptureState.IDLE, AutoCaptureState.STABILIZING)

        # 5. Steady again on new page -> Triggers next capture!
        controller.update(res_turned, q_pass=True, dt=0.3)
        captured, state, prog, msg = controller.update(res_turned, q_pass=True, dt=0.3)
        assert captured is True
        assert state == AutoCaptureState.TRIGGERED

    def test_manual_capture_notifications(self, controller, base_corners):
        """Manual capture (key 'S') puts controller into cooldown to avoid immediate duplicate."""
        controller.notify_manual_capture(base_corners)
        assert controller.state == AutoCaptureState.COOLDOWN
        assert controller.cooldown_timer == controller.config.cooldown_duration

    def test_single_corner_flex_does_not_trigger_page_turn(self, controller, base_corners):
        """Smoothing or flexing only one corner (e.g. 70px) while 3 corners stay still must NOT trigger page turn."""
        res = make_result(base_corners)

        # 1. Trigger capture
        controller.update(res, q_pass=True, dt=0.1)
        controller.update(res, q_pass=True, dt=0.3)
        captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.3)
        assert captured is True
        assert state == AutoCaptureState.TRIGGERED

        # 2. Advance through cooldown into WAITING_FOR_PAGE_TURN
        controller.update(res, q_pass=True, dt=0.1)
        controller.update(res, q_pass=True, dt=0.5)
        assert controller.state == AutoCaptureState.WAITING_FOR_PAGE_TURN

        # 3. Only corner 3 shifts by 70px (e.g. user smoothed down corner), others stay pinned (0px)
        flexed_corners = base_corners.copy()
        flexed_corners[3] += np.array([50.0, 50.0])  # ~70.7px displacement on single corner
        res_flexed = make_result(flexed_corners)

        captured, state, prog, msg = controller.update(res_flexed, q_pass=True, dt=0.1)
        assert captured is False
        # Must stay in WAITING_FOR_PAGE_TURN, NOT re-arm!
        assert state == AutoCaptureState.WAITING_FOR_PAGE_TURN
        assert controller.page_turn_detected is False

    def test_unrefined_edge_does_not_trigger(self, controller, base_corners):
        """Coarse YOLO_DIRECT detections must not fire auto-capture; must wait for refined edge."""
        res = make_result(base_corners)
        res.detection_method = DetectionMethod.YOLO_DIRECT

        for _ in range(5):
            captured, state, prog, msg = controller.update(res, q_pass=True, dt=0.2)
            assert captured is False
            assert "Aligning booklet edges" in msg
            assert controller.stable_timer == 0.0

    def test_post_hand_settle_delay(self, controller, base_corners):
        """After hands leave the booklet, a settle timer must elapse before countdown begins."""
        res_hands = make_result(base_corners, hands=True)
        res_clean = make_result(base_corners, hands=False)

        # Hand in frame
        controller.update(res_hands, q_pass=True, dt=0.1)
        assert controller.hand_settle_timer > 0.0

        # Hand removed: 1st frame during settle window (0.2s of 0.6s)
        captured, state, prog, msg = controller.update(res_clean, q_pass=True, dt=0.2)
        assert captured is False
        assert "settling" in msg.lower()
        assert controller.stable_timer == 0.0

        # Settle window finishes (advance 0.5s > 0.4s remaining)
        controller.update(res_clean, q_pass=True, dt=0.5)

        # Now stabilization starts (frame 1 registers initial corners)
        captured, state, prog, msg = controller.update(res_clean, q_pass=True, dt=0.1)
        assert state == AutoCaptureState.STABILIZING

        # Frame 2 accumulates stability timer
        captured, state, prog, msg = controller.update(res_clean, q_pass=True, dt=0.2)
        assert state == AutoCaptureState.STABILIZING
        assert controller.stable_timer > 0.0



