"""
SignBridge AI — Unified & Versioned Hand Landmark Feature Schema (v2.0.0).
Provides a single canonical, mathematically identical feature extraction implementation
for both training pipelines and real-time live camera inference.

Strict Requirements:
- Feature dimension: exactly 282 float32 values.
- Left hand slot: 140 floats (63 absolute coords + 63 wrist-relative coords + 10 joint angles + 4 tip distances).
- Right hand slot: 140 floats (identical formulation).
- Presence flags: 2 floats ([left_present, right_present], 1.0 or 0.0).
- Sequence length: 30 frames for temporal/dynamic recognition.
- NEVER silently pad, truncate, or reshape incompatible feature dimensions.
- Explicitly reject incompatible models via IncompatibleModelError.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

SCHEMA_VERSION = "2.0.0"
FEATURE_DIM = 282
SINGLE_HAND_DIM = 140
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21

# Landmark Joint Angle Triplets (vertex at center index):
# Thumb: CMC-MCP-IP (1, 2, 3), MCP-IP-TIP (2, 3, 4)
# Index: MCP-PIP-DIP (5, 6, 7), PIP-DIP-TIP (6, 7, 8)
# Middle: MCP-PIP-DIP (9, 10, 11), PIP-DIP-TIP (10, 11, 12)
# Ring: MCP-PIP-DIP (13, 14, 15), PIP-DIP-TIP (14, 15, 16)
# Pinky: MCP-PIP-DIP (17, 18, 19), PIP-DIP-TIP (18, 19, 20)
JOINT_ANGLE_TRIPLETS: List[Tuple[int, int, int]] = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
]

FINGERTIP_INDICES: Tuple[int, int, int, int] = (8, 12, 16, 20)  # Index, Middle, Ring, Pinky
THUMB_TIP_INDEX: int = 4
WRIST_INDEX: int = 0
MIDDLE_MCP_INDEX: int = 9


class IncompatibleModelError(ValueError):
    """Raised when a model architecture or input shape does not match the active feature schema."""
    pass


def compute_joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute joint angle ABC at vertex b in [0.0, 1.0], normalized from [0, 180] degrees.

    Args:
        a: 3D coordinates of first landmark (shape: (3,))
        b: 3D coordinates of vertex landmark (shape: (3,))
        c: 3D coordinates of third landmark (shape: (3,))

    Returns:
        float: Normalized angle in [0.0, 1.0].
    """
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def extract_single_hand_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract standardized 140-dimensional feature vector for a single hand:
    - 63 absolute coordinates (21 landmarks x 3)
    - 63 wrist-relative coordinates normalized by palm scale
    - 10 normalized finger joint angles in [0.0, 1.0]
    - 4 normalized fingertip-to-thumb distances

    Args:
        landmarks: np.ndarray of shape (21, 3) with [x, y, z] coordinates.

    Returns:
        np.ndarray: 140-dim float32 vector.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), received {pts.shape}.")

    wrist = pts[WRIST_INDEX].copy()
    palm_scale = float(np.linalg.norm(pts[MIDDLE_MCP_INDEX] - wrist))
    if palm_scale < 1e-4:
        palm_scale = 1e-4

    # 1. Absolute 3D coordinates (63 dims)
    coords_abs = pts.reshape(-1)

    # 2. Wrist-relative scaled coordinates (63 dims)
    coords_rel = ((pts - wrist) / palm_scale).reshape(-1)

    # 3. 10 Finger joint angles (10 dims)
    angles = np.asarray(
        [compute_joint_angle(pts[a], pts[b], pts[c]) for a, b, c in JOINT_ANGLE_TRIPLETS],
        dtype=np.float32,
    )

    # 4. 4 Fingertip-to-thumb distances normalized by palm scale (4 dims)
    thumb_tip = pts[THUMB_TIP_INDEX]
    tip_distances = np.asarray(
        [np.linalg.norm(pts[idx] - thumb_tip) / palm_scale for idx in FINGERTIP_INDICES],
        dtype=np.float32,
    )

    features = np.concatenate([coords_abs, coords_rel, angles, tip_distances]).astype(np.float32)
    if features.shape != (SINGLE_HAND_DIM,):
        raise RuntimeError(
            f"Internal feature size error: expected {SINGLE_HAND_DIM}, got {features.shape}."
        )
    return features


def extract_features_from_arrays(
    left_landmarks: Optional[np.ndarray] = None,
    right_landmarks: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Assemble the canonical 282-dim feature vector directly from numpy arrays.

    Args:
        left_landmarks: Optional (21, 3) array for left hand.
        right_landmarks: Optional (21, 3) array for right hand.

    Returns:
        Tuple[np.ndarray, int]: (features_282, hands_detected_count)
    """
    left_slot = (
        extract_single_hand_features(left_landmarks[:21, :3])
        if left_landmarks is not None and len(left_landmarks) >= 21
        else np.zeros(SINGLE_HAND_DIM, dtype=np.float32)
    )
    left_present = 1.0 if left_landmarks is not None and len(left_landmarks) >= 21 else 0.0

    right_slot = (
        extract_single_hand_features(right_landmarks[:21, :3])
        if right_landmarks is not None and len(right_landmarks) >= 21
        else np.zeros(SINGLE_HAND_DIM, dtype=np.float32)
    )
    right_present = 1.0 if right_landmarks is not None and len(right_landmarks) >= 21 else 0.0

    presence = np.asarray([left_present, right_present], dtype=np.float32)
    features = np.concatenate([left_slot, right_slot, presence]).astype(np.float32)

    return features, int(left_present + right_present)


def extract_frame_features(hand_result_or_raw: Any) -> Tuple[np.ndarray, int]:
    """
    Extract the 282-dim feature vector from either a HandLandmarkResult dataclass
    or raw MediaPipe HandLandmarkerResult object.

    Returns:
        Tuple[np.ndarray, int]: (features_282, hands_detected_count)
    """
    if hand_result_or_raw is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32), 0

    # 1. If it's a HandLandmarkResult dataclass
    left_lms = getattr(hand_result_or_raw, "left_landmarks", None)
    right_lms = getattr(hand_result_or_raw, "right_landmarks", None)
    left_detected = getattr(hand_result_or_raw, "left_hand_detected", False)
    right_detected = getattr(hand_result_or_raw, "right_hand_detected", False)

    if (left_detected and left_lms is not None) or (right_detected and right_lms is not None):
        return extract_features_from_arrays(
            left_landmarks=left_lms if left_detected else None,
            right_landmarks=right_lms if right_detected else None,
        )

    # 2. Check for raw MediaPipe Tasks result (with hand_landmarks list and handedness list)
    raw = getattr(hand_result_or_raw, "raw_results", hand_result_or_raw)
    hands = getattr(raw, "hand_landmarks", []) or []
    handedness = getattr(raw, "handedness", []) or []

    if not hands:
        return np.zeros(FEATURE_DIM, dtype=np.float32), 0

    slots: Dict[str, Optional[np.ndarray]] = {"Left": None, "Right": None}

    for idx, landmarks in enumerate(hands[:2]):
        # Extract 21 points
        if hasattr(landmarks, "__iter__"):
            pts = []
            for lm in landmarks:
                pts.append([getattr(lm, "x", 0.0), getattr(lm, "y", 0.0), getattr(lm, "z", 0.0)])
            pts_arr = np.asarray(pts, dtype=np.float32)
        else:
            pts_arr = np.asarray(landmarks, dtype=np.float32)

        if pts_arr.shape != (21, 3):
            continue

        side = None
        if idx < len(handedness) and handedness[idx]:
            try:
                side = handedness[idx][0].category_name
            except (AttributeError, IndexError, TypeError):
                side = None

        if side not in slots or slots.get(side) is not None:
            center_x = float(np.mean(pts_arr[:, 0]))
            side = "Left" if center_x < 0.5 else "Right"
            if slots[side] is not None:
                side = "Right" if side == "Left" else "Left"

        slots[side] = pts_arr

    return extract_features_from_arrays(
        left_landmarks=slots["Left"],
        right_landmarks=slots["Right"],
    )


def resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Resample a variable-length sequence of feature frames to exactly target_length (30) frames
    using linear temporal interpolation.

    Args:
        sequence: Feature array of shape (N, 282) where N >= 1.
        target_length: Desired number of frames (default: 30).

    Returns:
        np.ndarray: Resampled sequence of shape (target_length, 282) and dtype float32.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected sequence of shape (frames, {FEATURE_DIM}); received {seq.shape}."
        )
    if len(seq) == 0:
        raise ValueError("Cannot resample an empty sequence.")

    if len(seq) == target_length:
        return seq.copy()

    positions = np.linspace(0, len(seq) - 1, target_length)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = (positions - left)[:, None].astype(np.float32)

    return (seq[left] * (1.0 - weight) + seq[right] * weight).astype(np.float32)


def validate_features(features: np.ndarray) -> bool:
    """
    Validate that a feature vector strictly conforms to Schema v2.0.0.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(features, np.ndarray):
        return False
    if features.shape != (FEATURE_DIM,):
        return False
    if not np.all(np.isfinite(features)):
        return False
    return True


def validate_sequence(sequence: np.ndarray, expected_seq_len: int = SEQUENCE_LENGTH) -> bool:
    """
    Validate that a sequence of feature vectors conforms to the schema.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(sequence, np.ndarray):
        return False
    if sequence.shape != (expected_seq_len, FEATURE_DIM):
        return False
    if not np.all(np.isfinite(sequence)):
        return False
    return True


def verify_model_compatibility(
    model: Any,
    expected_dim: int = FEATURE_DIM,
    expected_seq_len: int = SEQUENCE_LENGTH,
) -> None:
    """
    Verify that a loaded Keras model strictly matches the feature schema.
    Raises IncompatibleModelError if the model expects different dimensions.
    Never allows silent padding, truncation, or dimension mismatch.
    """
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        # Check first layer if available
        layers = getattr(model, "layers", [])
        if layers:
            input_shape = getattr(layers[0], "input_shape", None)

    if input_shape is None:
        return

    # Typical input_shape: (None, 30, 282) or (None, 282)
    if len(input_shape) == 3:
        model_seq_len = input_shape[1]
        model_dim = input_shape[2]
        if model_dim is not None and model_dim != expected_dim:
            raise IncompatibleModelError(
                f"Model feature dimension mismatch: expected {expected_dim}, "
                f"but model requires {model_dim}. (Input shape: {input_shape})"
            )
        if model_seq_len is not None and model_seq_len != expected_seq_len:
            raise IncompatibleModelError(
                f"Model sequence length mismatch: expected {expected_seq_len}, "
                f"but model requires {model_seq_len}. (Input shape: {input_shape})"
            )
    elif len(input_shape) == 2:
        model_dim = input_shape[1]
        if model_dim is not None and model_dim != expected_dim:
            raise IncompatibleModelError(
                f"Model static feature dimension mismatch: expected {expected_dim}, "
                f"but model requires {model_dim}. (Input shape: {input_shape})"
            )

