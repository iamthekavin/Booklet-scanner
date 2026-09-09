"""
Auto-Capture Controller for VEE Scanner.

Provides automated, hands-free booklet capture with:
1. Corner stability tracking across frames.
2. Quality and hand-occlusion validation.
3. Configurable steady countdown with visual progress.
4. Anti-duplicate re-arm state machine (cooldown + page turn detection).
5. Operator toggle (key 'A').
"""

import enum
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from models import DetectionResult, ReviewFlag

logger = logging.getLogger(__name__)


class AutoCaptureState(enum.Enum):
    DISABLED = "disabled"
    IDLE = "idle"
    STABILIZING = "stabilizing"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"
    WAITING_FOR_PAGE_TURN = "waiting_page_turn"


@dataclass
class AutoCaptureConfig:
    stability_duration: float = 1.0     # Seconds booklet must remain stationary
    drift_threshold_px: float = 14.0    # Maximum corner displacement allowed per frame (px)
    min_confidence: float = 0.45        # Minimum detector confidence to allow capture
    cooldown_duration: float = 1.8      # Post-capture pause before looking for page turn (s)
    page_turn_drift_px: float = 35.0    # Corner displacement threshold indicating page has turned (px)
    require_no_hands: bool = True       # Require hands to be absent before triggering


class AutoCaptureController:
    """Manages the auto-capture lifecycle for hands-free scanning."""

    def __init__(
        self,
        config: Optional[AutoCaptureConfig] = None,
        enabled: bool = True,
    ) -> None:
        self.config = config or AutoCaptureConfig()
        self.enabled = enabled
        self.state = AutoCaptureState.IDLE if enabled else AutoCaptureState.DISABLED

        self.stable_timer: float = 0.0
        self.cooldown_timer: float = 0.0
        self.last_corners: Optional[np.ndarray] = None
        self.captured_corners: Optional[np.ndarray] = None
        self.page_turn_detected: bool = False
        self.last_status: str = "Ready"

    def toggle(self) -> bool:
        """Toggle auto-capture ON/OFF."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.state = AutoCaptureState.DISABLED
            self.reset()
        else:
            self.state = AutoCaptureState.IDLE
            self.reset()
        return self.enabled

    def enable(self) -> None:
        """Enable auto-capture."""
        if not self.enabled:
            self.toggle()

    def disable(self) -> None:
        """Disable auto-capture."""
        if self.enabled:
            self.toggle()

    def reset(self) -> None:
        """Reset internal timers and tracking variables."""
        self.stable_timer = 0.0
        self.cooldown_timer = 0.0
        self.last_corners = None
        self.captured_corners = None
        self.page_turn_detected = False
        if self.enabled:
            self.state = AutoCaptureState.IDLE

    def notify_manual_capture(self, corners: Optional[np.ndarray] = None) -> None:
        """Inform the controller that a manual capture occurred (e.g. key 'S')."""
        if corners is not None:
            self.captured_corners = corners.copy()
        self.cooldown_timer = self.config.cooldown_duration
        if self.enabled:
            self.state = AutoCaptureState.COOLDOWN
        self.stable_timer = 0.0
        self.page_turn_detected = False

    def update(
        self,
        result: Optional[DetectionResult],
        q_pass: bool = True,
        dt: float = 0.033,
    ) -> Tuple[bool, AutoCaptureState, float, str]:
        """Update auto-capture state with the current frame's detection.

        Args:
            result: Current detection result from BookletDetector.
            q_pass: True if frame passed the quality gate (sharpness, glare).
            dt: Elapsed time in seconds since last frame.

        Returns:
            Tuple of (should_capture, state, progress_0_to_1, status_message)
        """
        if not self.enabled:
            self.state = AutoCaptureState.DISABLED
            return False, self.state, 0.0, "Auto: OFF [A]"

        # Transition from TRIGGERED -> COOLDOWN
        if self.state == AutoCaptureState.TRIGGERED:
            self.state = AutoCaptureState.COOLDOWN
            self.cooldown_timer = self.config.cooldown_duration

        # ── State: COOLDOWN ──────────────────────────────────────────
        if self.state == AutoCaptureState.COOLDOWN:
            self.cooldown_timer -= dt
            if self.cooldown_timer <= 0:
                self.state = AutoCaptureState.WAITING_FOR_PAGE_TURN
                self.last_status = "Turn page..."
            else:
                progress = max(0.0, self.cooldown_timer / self.config.cooldown_duration)
                return False, self.state, progress, f"Captured! ({self.cooldown_timer:.1f}s)"

        # ── State: WAITING_FOR_PAGE_TURN ─────────────────────────────
        if self.state == AutoCaptureState.WAITING_FOR_PAGE_TURN:
            has_corners = result is not None and result.corners is not None

            # Case 1: Booklet temporarily removed / occluded during page turn
            if not has_corners:
                self.page_turn_detected = True

            # Case 2: Hand interaction detected (flipping / adjusting page)
            elif result is not None and (bool(result.hands_detected) or (ReviewFlag.HAND_OCCLUSION in result.review_flags)):
                self.page_turn_detected = True

            # Case 3: Significant booklet movement / geometry change
            elif self.captured_corners is not None:
                curr_pts = result.corners.points.astype(np.float32)
                corner_dists = np.linalg.norm(curr_pts - self.captured_corners, axis=1)
                mean_dist = float(np.mean(corner_dists))
                max_dist = float(np.max(corner_dists))
                num_shifted = int(np.sum(corner_dists >= 20.0))

                # Page turn / booklet repositioning requires either:
                # - Broad movement across the entire quad (mean_dist >= drift threshold), OR
                # - At least two corners shifted significantly (prevents single-corner flex/flutter false triggers)
                is_movement = (mean_dist >= self.config.page_turn_drift_px) or (
                    num_shifted >= 2 and max_dist >= (self.config.page_turn_drift_px * 1.5)
                )
                if is_movement:
                    self.page_turn_detected = True

            # If page turn was detected and booklet is now detected again, re-arm!
            if self.page_turn_detected and has_corners:
                self.state = AutoCaptureState.IDLE
                self.page_turn_detected = False
                self.stable_timer = 0.0
                self.last_corners = None
            else:
                return False, self.state, 0.0, "Turn to next page..."

        # ── Check Prerequisites for IDLE / STABILIZING ──────────────
        if not q_pass:
            self.stable_timer = 0.0
            self.state = AutoCaptureState.IDLE
            return False, self.state, 0.0, "Stabilizing camera..."

        if result is None or result.corners is None:
            self.stable_timer = 0.0
            self.state = AutoCaptureState.IDLE
            self.last_corners = None
            return False, self.state, 0.0, "Waiting for booklet..."

        if result.confidence < self.config.min_confidence:
            self.stable_timer = 0.0
            self.state = AutoCaptureState.IDLE
            return False, self.state, 0.0, "Low confidence detection"

        if self.config.require_no_hands:
            has_hands = bool(result.hands_detected) or (ReviewFlag.HAND_OCCLUSION in result.review_flags)
            if has_hands:
                self.stable_timer = 0.0
                self.state = AutoCaptureState.IDLE
                return False, self.state, 0.0, "Hand detected - remove hand"

        # ── Stability Tracking ───────────────────────────────────────
        curr_corners = result.corners.points.astype(np.float32)

        if self.last_corners is None:
            self.last_corners = curr_corners
            self.stable_timer = 0.0
            self.state = AutoCaptureState.STABILIZING
            return False, self.state, 0.0, "Hold still..."

        # Calculate max displacement among all 4 corners
        drift = float(np.max(np.linalg.norm(curr_corners - self.last_corners, axis=1)))
        self.last_corners = curr_corners

        if drift > self.config.drift_threshold_px:
            # Significant movement: reduce or reset stability timer
            self.stable_timer = max(0.0, self.stable_timer - dt * 2.0)
            self.state = AutoCaptureState.STABILIZING
            progress = min(1.0, self.stable_timer / self.config.stability_duration)
            return False, self.state, progress, "Hold still..."
        else:
            # Steady! Accumulate stability duration
            self.stable_timer += dt
            progress = min(1.0, self.stable_timer / self.config.stability_duration)

            if self.stable_timer >= self.config.stability_duration:
                # Target duration reached -> Trigger capture!
                self.state = AutoCaptureState.TRIGGERED
                self.captured_corners = curr_corners.copy()
                self.stable_timer = 0.0
                logger.info("AutoCapture triggered!")
                return True, self.state, 1.0, "CAPTURING!"
            else:
                self.state = AutoCaptureState.STABILIZING
                pct = int(progress * 100)
                return False, self.state, progress, f"Hold steady ({pct}%)"
