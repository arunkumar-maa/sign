"""
Camera & Finger Detection Diagnostics Engine for SignBridge AI.
Evaluates camera feed quality, lighting, tracking metrics, and tests 9 specific
diagnostic hand postures (Open palm, Closed fist, Index, Two fingers, Three fingers,
Thumb, Sideways hand, Both hands, Moving hand) to benchmark tracking limits.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from backend.vision.hand_landmarks import HandLandmarkResult, FINGERTIP_INDICES

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticMetrics:
    """Real-time camera and tracking diagnostic telemetry."""
    camera_active: bool = False
    fps: float = 0.0
    inference_latency_ms: float = 0.0
    lighting_condition: str = "Good"   # "Good", "Low brightness (dark)", "Overexposed"
    mean_brightness: float = 0.0
    num_hands: int = 0
    total_landmarks: int = 0
    left_hand_active: bool = False
    right_hand_active: bool = False
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0
    tracking_quality: str = "No hands detected"
    active_posture: str = "None"
    posture_test_results: Dict[str, bool] = field(default_factory=dict)

    def summary_text(self) -> str:
        lines = [
            f"📷 Camera Status: {'Active (Connected)' if self.camera_active else 'Disconnected / Inactive'}",
            f"⚡ FPS: {self.fps:.1f} | Latency: {self.inference_latency_ms:.1f} ms",
            f"💡 Lighting: {self.lighting_condition} (Mean: {self.mean_brightness:.1f}/255)",
            f"🖐 Hands Detected: {self.num_hands} | Total Landmarks: {self.total_landmarks}",
            f"  • Left Hand:  {'✓' if self.left_hand_active else '✗'} ({self.left_hand_confidence:.0%})",
            f"  • Right Hand: {'✓' if self.right_hand_active else '✗'} ({self.right_hand_confidence:.0%})",
            f"📊 Tracking Quality: {self.tracking_quality}",
            f"🎯 Detected Posture: {self.active_posture}",
        ]
        return "\n".join(lines)


class FingerDiagnostics:
    """
    Evaluates 9 specific hand diagnostic postures and camera tracking conditions.
    """

    POSTURE_NAMES = [
        "1. Open palm",
        "2. Closed fist",
        "3. Index finger",
        "4. Two fingers",
        "5. Three fingers",
        "6. Thumb",
        "7. Sideways hand",
        "8. Both hands",
        "9. Moving hand",
    ]

    def __init__(self):
        self._prev_wrist_pos = None
        self._prev_time = time.time()
        self._velocity_history: List[float] = []
        self._posture_passes: Dict[str, bool] = {name: False for name in self.POSTURE_NAMES}

    def reset_tests(self):
        self._posture_passes = {name: False for name in self.POSTURE_NAMES}
        self._velocity_history.clear()

    def analyze_posture(
        self,
        hand_result: HandLandmarkResult,
        frame_shape: Tuple[int, int],
    ) -> Tuple[str, Dict[str, bool]]:
        """
        Classify the active diagnostic hand posture and update test achievements.

        Args:
            hand_result: Result from HandLandmarker.
            frame_shape: (height, width) of the image.

        Returns:
            active_posture_name, dictionary of {posture_name: passed_bool}
        """
        if not hand_result.detected:
            return "None", dict(self._posture_passes)

        # 8. Check Both Hands
        if hand_result.left_hand_detected and hand_result.right_hand_detected:
            self._posture_passes["8. Both hands"] = True

        # Use primary available hand (prefer right hand or left hand)
        lms = hand_result.right_landmarks if hand_result.right_hand_detected else hand_result.left_landmarks
        if lms is None or len(lms) < 21:
            return "Incomplete landmarks", dict(self._posture_passes)

        wrist = lms[0]
        curr_time = time.time()
        dt = curr_time - self._prev_time if (curr_time - self._prev_time) > 0.001 else 0.033
        self._prev_time = curr_time

        # Calculate hand motion speed
        if self._prev_wrist_pos is not None:
            dist = np.linalg.norm(wrist[:2] - self._prev_wrist_pos[:2])
            velocity = dist / dt
            self._velocity_history.append(velocity)
            if len(self._velocity_history) > 10:
                self._velocity_history.pop(0)

            # 9. Moving Hand (velocity above threshold)
            if np.mean(self._velocity_history) > 0.8:
                self._posture_passes["9. Moving hand"] = True
        self._prev_wrist_pos = wrist.copy()

        # Check Sideways Hand (Angle between wrist and middle MCP)
        middle_mcp = lms[9]
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        # In sideways orientation, horizontal displacement dominates vertical
        if abs(dx) > abs(dy) * 1.3:
            self._posture_passes["7. Sideways hand"] = True

        # Compute finger extension states (Thumb, Index, Middle, Ring, Pinky)
        # Using distance from wrist to tip vs wrist to PIP/MCP
        d_wrist_mcp9 = np.linalg.norm(middle_mcp - wrist)
        if d_wrist_mcp9 < 1e-5:
            d_wrist_mcp9 = 0.1

        # Index (8 vs 6)
        index_ext = np.linalg.norm(lms[8] - wrist) > np.linalg.norm(lms[6] - wrist) * 1.15
        # Middle (12 vs 10)
        middle_ext = np.linalg.norm(lms[12] - wrist) > np.linalg.norm(lms[10] - wrist) * 1.15
        # Ring (16 vs 14)
        ring_ext = np.linalg.norm(lms[16] - wrist) > np.linalg.norm(lms[14] - wrist) * 1.15
        # Pinky (20 vs 18)
        pinky_ext = np.linalg.norm(lms[20] - wrist) > np.linalg.norm(lms[18] - wrist) * 1.15
        # Thumb (4 vs pinky MCP 17)
        thumb_ext = np.linalg.norm(lms[4] - lms[17]) > np.linalg.norm(lms[2] - lms[17]) * 1.25

        active = "Unclassified handshape"

        # 1. Open Palm: all 5 extended
        if index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext:
            active = "Open palm"
            self._posture_passes["1. Open palm"] = True

        # 2. Closed Fist: all 5 folded
        elif not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
            active = "Closed fist"
            self._posture_passes["2. Closed fist"] = True

        # 3. Index finger only
        elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
            active = "Index finger"
            self._posture_passes["3. Index finger"] = True

        # 4. Two fingers: index + middle
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            active = "Two fingers (Victory/V)"
            self._posture_passes["4. Two fingers"] = True

        # 5. Three fingers: index + middle + ring
        elif index_ext and middle_ext and ring_ext and not pinky_ext:
            active = "Three fingers"
            self._posture_passes["5. Three fingers"] = True

        # 6. Thumb only
        elif thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            active = "Thumb extended"
            self._posture_passes["6. Thumb"] = True

        elif hand_result.left_hand_detected and hand_result.right_hand_detected:
            active = "Both hands"

        return active, dict(self._posture_passes)

    def evaluate_frame(
        self,
        frame: np.ndarray,
        hand_result: HandLandmarkResult,
        fps: float,
    ) -> DiagnosticMetrics:
        """
        Evaluate full camera frame metrics and tracking quality.
        """
        if frame is None or frame.size == 0:
            return DiagnosticMetrics()

        # Lighting check via grayscale luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_lum = float(np.mean(gray))

        if mean_lum < 50.0:
            lighting = "Low brightness (Too Dark)"
        elif mean_lum > 225.0:
            lighting = "Overexposed (Too Bright)"
        else:
            lighting = "Good"

        active_posture, test_passes = self.analyze_posture(hand_result, frame.shape[:2])

        total_lms = (21 if hand_result.left_hand_detected else 0) + (21 if hand_result.right_hand_detected else 0)

        # Overall tracking quality assessment
        if hand_result.left_hand_detected and hand_result.right_hand_detected:
            quality = "Excellent (Both hands tracked)"
        elif hand_result.detected:
            quality = "Good (Single hand tracked)"
        elif mean_lum < 50.0:
            quality = "Degraded (Poor lighting hinders tracking)"
        else:
            quality = "Waiting for hands in frame"

        return DiagnosticMetrics(
            camera_active=True,
            fps=fps,
            inference_latency_ms=hand_result.inference_latency_ms,
            lighting_condition=lighting,
            mean_brightness=mean_lum,
            num_hands=hand_result.num_hands,
            total_landmarks=total_lms,
            left_hand_active=hand_result.left_hand_detected,
            right_hand_active=hand_result.right_hand_detected,
            left_hand_confidence=hand_result.left_confidence,
            right_hand_confidence=hand_result.right_confidence,
            tracking_quality=quality,
            active_posture=active_posture,
            posture_test_results=test_passes,
        )

