"""
SignBridge AI — Phase 2: Invariant 3D Finger Counter.
Classifies exact count of extended fingers (0, 1, 2, 3, 4, 5) using MediaPipe 21 hand landmarks.
Strictly robust to hand rotation, tilting, in-plane and out-of-plane orientation.
Does NOT rely on comparing 2D Y coordinates, but on 3D joint angles, distance ratios,
and palm-relative kinematics.

Diagnostic tool: Provides geometric finger count. Separate from ISL number signs.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from backend.vision.hand_pose_features import (
    calculate_angle_3d,
    WRIST,
    THUMB_CMC,
    THUMB_MCP,
    THUMB_IP,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_DIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_DIP,
    MIDDLE_TIP,
    RING_MCP,
    RING_PIP,
    RING_DIP,
    RING_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    PINKY_TIP,
)

logger = logging.getLogger(__name__)


@dataclass
class FingerCountResult:
    """Output from 3D geometric finger count classifier."""
    count: int = 0
    hand_detected: bool = False
    handedness: str = "Right"
    confidence: float = 0.0
    status: str = "No hand detected"
    extended_fingers: Dict[str, bool] = field(default_factory=lambda: {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    })
    angles: Dict[str, float] = field(default_factory=dict)
    ratios: Dict[str, float] = field(default_factory=dict)

    @property
    def details(self) -> Dict[str, Any]:
        ext_names = [f.capitalize() for f, is_ext in self.extended_fingers.items() if is_ext]
        return {
            "fingers": ", ".join(ext_names) if ext_names else "Folded",
            "extended": self.extended_fingers,
            "angles": self.angles,
            "ratios": self.ratios,
        }

    def summary(self) -> str:
        if not self.hand_detected:
            return "Finger Count: None (No hand detected)"
        ext_names = [f.capitalize() for f, is_ext in self.extended_fingers.items() if is_ext]
        ext_str = ", ".join(ext_names) if ext_names else "All fingers folded"
        return f"Finger Count: {self.count} [{ext_str}] (Quality: {self.confidence:.0%})"


class FingerCounter:
    """
    3D Rotation-Invariant Finger Counter.
    Analyzes 21 MediaPipe hand landmarks using intrinsic 3D angles and scale-normalized ratios.
    """

    def __init__(
        self,
        min_angle_extended: float = 2.25,   # ~129 degrees straight joint threshold
        min_distance_ratio: float = 1.18,   # Tip-to-wrist / PIP-to-wrist ratio
    ):
        self.min_angle_extended = min_angle_extended
        self.min_distance_ratio = min_distance_ratio

    def count_fingers(
        self,
        landmarks: Optional[np.ndarray],
        handedness: str = "Right",
    ) -> FingerCountResult:
        """
        Count extended fingers for a hand of 21 landmarks.

        Args:
            landmarks: np.ndarray of shape (21, 3) or (21, 2)
            handedness: 'Left' or 'Right'

        Returns:
            FingerCountResult with count (0..5), confidence, and extended digit details.
        """
        if landmarks is None or len(landmarks) < 21:
            return FingerCountResult()

        lms = np.asarray(landmarks, dtype=np.float32)[:21, :3]
        wrist = lms[WRIST]

        # 1. 4 Fingers Analysis (Index, Middle, Ring, Pinky)
        finger_specs = [
            ("index", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            ("ring", RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            ("pinky", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ]

        extended_flags: Dict[str, bool] = {}
        angles_record: Dict[str, float] = {}
        ratios_record: Dict[str, float] = {}
        confidences: List[float] = []

        for name, mcp_idx, pip_idx, dip_idx, tip_idx in finger_specs:
            mcp = lms[mcp_idx]
            pip = lms[pip_idx]
            dip = lms[dip_idx]
            tip = lms[tip_idx]

            # Invariant 3D joint angles
            angle_pip = calculate_angle_3d(mcp, pip, dip)
            angle_dip = calculate_angle_3d(pip, dip, tip)
            angles_record[f"{name}_pip"] = float(angle_pip)
            angles_record[f"{name}_dip"] = float(angle_dip)

            # Invariant Euclidean distance from wrist
            dist_tip = float(np.linalg.norm(tip - wrist))
            dist_pip = float(np.linalg.norm(pip - wrist))
            ratio = dist_tip / (dist_pip + 1e-6)
            ratios_record[f"{name}_ratio"] = ratio

            # Extension criteria:
            # 1. Angle at PIP must be relatively straight (> min_angle_extended)
            # 2. Tip must be substantially further from wrist than PIP
            is_extended = (angle_pip >= self.min_angle_extended and ratio >= self.min_distance_ratio)
            # Strong fold condition: curled into palm
            if ratio < 0.98 or angle_pip < 1.85:
                is_extended = False

            extended_flags[name] = bool(is_extended)

            # Compute quality confidence
            if is_extended:
                conf = min(1.0, (ratio - 1.0) / 0.4) * min(1.0, angle_pip / math.pi)
            else:
                conf = min(1.0, (1.2 - ratio) / 0.3)
            confidences.append(max(0.4, float(conf)))

        # 2. Thumb Analysis (Unique kinematics)
        # Thumb joints: Wrist (0), CMC (1), MCP (2), IP (3), TIP (4)
        thumb_tip = lms[THUMB_TIP]
        thumb_ip = lms[THUMB_IP]
        thumb_mcp = lms[THUMB_MCP]
        thumb_cmc = lms[THUMB_CMC]
        pinky_mcp = lms[PINKY_MCP]
        index_mcp = lms[INDEX_MCP]

        angle_thumb_ip = calculate_angle_3d(thumb_mcp, thumb_ip, thumb_tip)
        angles_record["thumb_ip"] = float(angle_thumb_ip)

        # Distance from thumb tip to pinky base vs thumb IP to pinky base
        dist_thumb_tip_pinky = float(np.linalg.norm(thumb_tip - pinky_mcp))
        dist_thumb_ip_pinky = float(np.linalg.norm(thumb_ip - pinky_mcp))
        thumb_spread_ratio = dist_thumb_tip_pinky / (dist_thumb_ip_pinky + 1e-6)
        ratios_record["thumb_spread"] = thumb_spread_ratio

        # Distance from thumb tip to index MCP base (tuck check)
        dist_thumb_tip_index_mcp = float(np.linalg.norm(thumb_tip - index_mcp))
        palm_scale = float(np.linalg.norm(lms[MIDDLE_MCP] - wrist))
        thumb_tuck_norm = dist_thumb_tip_index_mcp / (palm_scale + 1e-6)
        ratios_record["thumb_tuck"] = thumb_tuck_norm

        # Extended criteria:
        # Straight IP angle and spread away from pinky base, not tucked tightly to index base
        thumb_extended = (
            angle_thumb_ip >= 2.35
            and thumb_spread_ratio >= 1.15
            and thumb_tuck_norm >= 0.40
        )
        # If thumb tip is tucked close across palm (< 0.35 palm scale), it is folded
        if thumb_tuck_norm < 0.32 or thumb_spread_ratio < 0.95:
            thumb_extended = False

        extended_flags["thumb"] = bool(thumb_extended)
        confidences.append(0.85 if thumb_extended else 0.80)

        # 3. Overall Count & Result Synthesis
        total_count = sum(1 for is_ext in extended_flags.values() if is_ext)
        overall_conf = float(np.mean(confidences))

        ext_names = [f.capitalize() for f, is_ext in extended_flags.items() if is_ext]
        ext_desc = ", ".join(ext_names) if ext_names else "None (Closed fist)"
        status_msg = f"{total_count} finger{'s' if total_count != 1 else ''} extended ({ext_desc})"

        return FingerCountResult(
            count=total_count,
            hand_detected=True,
            handedness=handedness,
            confidence=round(overall_conf, 3),
            status=status_msg,
            extended_fingers=extended_flags,
            angles=angles_record,
            ratios=ratios_record,
        )
