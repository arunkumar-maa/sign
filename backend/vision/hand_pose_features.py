"""
SignBridge AI — Hand Pose Feature Extraction & Geometric Analysis.
Extracts hand-pose based features directly from 21 MediaPipe hand landmarks.
Strictly focuses on hands and fingers — NO full body pose, shoulder, elbow, or face mesh dependencies.

Features computed per hand (104 dimensions):
1. Normalized 3D Coordinates (21 landmarks * 3 = 63 dims):
   - Translation centered at wrist (landmark 0).
   - Scale-normalized by palm dimension (distance from wrist to middle MCP).
2. Finger Joint Angles (15 dims):
   - 3 angles per finger (MCP, PIP, DIP) computed via vector dot products.
3. Finger Distances (12 dims):
   - Thumb tip to all 4 other fingertips.
   - Adjacent fingertip distances (index-middle, middle-ring, ring-pinky).
   - Fingertip to wrist distances for all 5 fingers.
4. Handshape Extension & State (5 dims):
   - Extended, bent, or folded state for each finger [1.0, 0.5, 0.0].
5. Palm Orientation & Geometry (9 dims):
   - Palm normal 3D unit vector (cross product of palm axes).
   - Palm pointing 3D unit direction (wrist to middle MCP).
   - Original wrist 3D coordinate (for spatial tracking).

Two-Hand Unified Feature Vector (216 dimensions):
- Left hand features (104 dims, or zeros if missing)
- Right hand features (104 dims, or zeros if missing)
- Hand presence flags (2 dims: is_left_present, is_right_present)
- Relative two-hand spatial features (6 dims: inter-wrist displacement vector,
  inter-wrist distance, min fingertip distance, palm normal alignment dot product).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Constant Indices for MediaPipe 21 Hand Landmarks
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGERTIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCP_INDICES = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
PIP_INDICES = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

SINGLE_HAND_FEATURE_DIM = 104
TWO_HAND_FEATURE_DIM = 216  # 104 * 2 + 2 (presence) + 6 (relative)


def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute 3D joint angle at vertex p2 in radians between vectors (p1 - p2) and (p3 - p2).

    Args:
        p1: Point 1 (shape (3,))
        p2: Vertex point (shape (3,))
        p3: Point 3 (shape (3,))

    Returns:
        Angle in radians [0.0, pi].
    """
    v1 = np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
    v2 = np.asarray(p3, dtype=np.float32) - np.asarray(p2, dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def normalize_hand_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Translate landmarks to wrist origin (landmark 0) and scale by palm size.

    Args:
        landmarks: shape (21, 3)

    Returns:
        Tuple of (normalized_landmarks shape (21, 3), scale_factor float)
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros((21, 3), dtype=np.float32), 1.0

    lms = np.asarray(landmarks, dtype=np.float32)[:21, :3].copy()
    wrist = lms[WRIST].copy()

    # Center at wrist
    centered = lms - wrist

    # Scale using distance from wrist to middle finger MCP
    palm_size = float(np.linalg.norm(centered[MIDDLE_MCP]))
    if palm_size < 1e-5:
        # Fallback to max distance
        palm_size = float(np.max(np.linalg.norm(centered, axis=1)))
    if palm_size < 1e-5:
        palm_size = 1.0

    normalized = centered / palm_size
    return normalized, palm_size


def calculate_finger_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Calculate 15 finger joint angles (3 per finger).

    Thumb:
      1. Wrist -> Thumb CMC -> Thumb MCP
      2. Thumb CMC -> Thumb MCP -> Thumb IP
      3. Thumb MCP -> Thumb IP -> Thumb TIP
    Fingers (Index, Middle, Ring, Pinky):
      1. Wrist -> MCP -> PIP
      2. MCP -> PIP -> DIP
      3. PIP -> DIP -> TIP

    Returns:
        np.ndarray of shape (15,) with angles in radians.
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(15, dtype=np.float32)

    lms = landmarks[:21, :3]
    angles = []

    # Thumb
    angles.append(calculate_angle_3d(lms[WRIST], lms[THUMB_CMC], lms[THUMB_MCP]))
    angles.append(calculate_angle_3d(lms[THUMB_CMC], lms[THUMB_MCP], lms[THUMB_IP]))
    angles.append(calculate_angle_3d(lms[THUMB_MCP], lms[THUMB_IP], lms[THUMB_TIP]))

    # 4 Fingers: Index, Middle, Ring, Pinky
    finger_triplets = [
        (WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]

    for wrist, mcp, pip, dip, tip in finger_triplets:
        angles.append(calculate_angle_3d(lms[wrist], lms[mcp], lms[pip]))
        angles.append(calculate_angle_3d(lms[mcp], lms[pip], lms[dip]))
        angles.append(calculate_angle_3d(lms[pip], lms[dip], lms[tip]))

    return np.array(angles, dtype=np.float32)


def calculate_finger_distances(landmarks: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Calculate 12 scale-normalized finger distance metrics:
    - Thumb tip to index, middle, ring, pinky tips (4 dims)
    - Adjacent fingertip distances (index-middle, middle-ring, ring-pinky) (3 dims)
    - Fingertip to wrist distances for all 5 fingers (5 dims)

    Returns:
        np.ndarray of shape (12,)
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(12, dtype=np.float32)

    s = max(scale, 1e-5)
    lms = landmarks[:21, :3]
    tt = lms[THUMB_TIP]
    it = lms[INDEX_TIP]
    mt = lms[MIDDLE_TIP]
    rt = lms[RING_TIP]
    pt = lms[PINKY_TIP]
    w = lms[WRIST]

    distances = [
        # Thumb to other tips
        np.linalg.norm(tt - it) / s,
        np.linalg.norm(tt - mt) / s,
        np.linalg.norm(tt - rt) / s,
        np.linalg.norm(tt - pt) / s,
        # Adjacent tips
        np.linalg.norm(it - mt) / s,
        np.linalg.norm(mt - rt) / s,
        np.linalg.norm(rt - pt) / s,
        # Tips to wrist
        np.linalg.norm(tt - w) / s,
        np.linalg.norm(it - w) / s,
        np.linalg.norm(mt - w) / s,
        np.linalg.norm(rt - w) / s,
        np.linalg.norm(pt - w) / s,
    ]
    return np.array(distances, dtype=np.float32)


def classify_finger_states(landmarks: np.ndarray) -> np.ndarray:
    """
    Determine extension state for each of the 5 fingers:
    1.0 = Extended
    0.5 = Bent / Curved
    0.0 = Folded / Curled into palm

    Returns:
        np.ndarray of shape (5,) for [Thumb, Index, Middle, Ring, Pinky]
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(5, dtype=np.float32)

    lms = landmarks[:21, :3]
    states = np.zeros(5, dtype=np.float32)
    wrist = lms[WRIST]

    # Thumb extension: test distance from tip to pinky MCP and angle at IP
    thumb_tip_dist = np.linalg.norm(lms[THUMB_TIP] - lms[PINKY_MCP])
    thumb_ip_dist = np.linalg.norm(lms[THUMB_IP] - lms[PINKY_MCP])
    thumb_angle = calculate_angle_3d(lms[THUMB_MCP], lms[THUMB_IP], lms[THUMB_TIP])

    if thumb_tip_dist > thumb_ip_dist * 1.15 and thumb_angle > 2.4:
        states[0] = 1.0  # Extended
    elif thumb_tip_dist > thumb_ip_dist * 0.9:
        states[0] = 0.5  # Bent
    else:
        states[0] = 0.0  # Folded

    # 4 Fingers: test tip distance to wrist compared to PIP distance to wrist
    finger_indices = [
        (INDEX_TIP, INDEX_PIP, INDEX_DIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_MCP),
        (RING_TIP, RING_PIP, RING_DIP, RING_MCP),
        (PINKY_TIP, PINKY_PIP, PINKY_DIP, PINKY_MCP),
    ]

    for idx, (tip, pip, dip, mcp) in enumerate(finger_indices, start=1):
        dist_tip_wrist = np.linalg.norm(lms[tip] - wrist)
        dist_pip_wrist = np.linalg.norm(lms[pip] - wrist)
        dist_dip_wrist = np.linalg.norm(lms[dip] - wrist)

        # Ratio of tip distance to pip distance
        ratio = dist_tip_wrist / (dist_pip_wrist + 1e-6)
        pip_angle = calculate_angle_3d(lms[mcp], lms[pip], lms[dip])

        if ratio > 1.25 and pip_angle > 2.2:
            states[idx] = 1.0  # Extended
        elif ratio > 0.95 or dist_tip_wrist > dist_dip_wrist:
            states[idx] = 0.5  # Bent
        else:
            states[idx] = 0.0  # Folded

    return states


def classify_handshape_name(landmarks: np.ndarray, is_left: bool = False) -> str:
    """
    Classify human-readable handshape name (e.g. Open Palm, Fist, Pointing, Pinch, Victory).

    Returns:
        Descriptive string representation of the handshape.
    """
    if landmarks is None or len(landmarks) < 21:
        return "None"

    states = classify_finger_states(landmarks)
    thumb, index, middle, ring, pinky = states

    # 1. Fist: all fingers folded (checked first before pinch)
    if thumb < 0.6 and index < 0.5 and middle < 0.5 and ring < 0.5 and pinky < 0.5:
        return "Fist"

    # Pinch check: thumb tip close to index tip
    thumb_index_dist = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])
    palm_scale = np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])
    is_pinch = (thumb_index_dist / (palm_scale + 1e-6)) < 0.35

    # OK Sign: thumb and index touching, other 3 extended
    if is_pinch and middle > 0.7 and ring > 0.7 and pinky > 0.7:
        return "OK Sign"

    # Pinch: thumb and index touching, other fingers folded or bent
    if is_pinch and middle < 0.6 and ring < 0.6 and pinky < 0.6 and (thumb >= 0.4 or index >= 0.4):
        return "Pinch"

    # Open Palm: all 5 fingers extended
    if thumb > 0.7 and index > 0.7 and middle > 0.7 and ring > 0.7 and pinky > 0.7:
        return "Open Palm"

    # Flat Hand: 4 fingers extended together
    if index > 0.7 and middle > 0.7 and ring > 0.7 and pinky > 0.7 and thumb <= 0.7:
        return "Flat Hand"

    # Thumbs Up: thumb extended upwards, other 4 folded
    if thumb > 0.7 and index < 0.5 and middle < 0.5 and ring < 0.5 and pinky < 0.5:
        # Check if thumb tip is above wrist
        if landmarks[THUMB_TIP][1] < landmarks[WRIST][1]:
            return "Thumbs Up"
        return "Thumbs Down"

    # Pointing / Index: index extended, others folded
    if index > 0.7 and middle < 0.5 and ring < 0.5 and pinky < 0.5:
        return "Pointing"

    # Victory / Peace: index and middle extended, others folded
    if index > 0.7 and middle > 0.7 and ring < 0.5 and pinky < 0.5:
        return "Victory / V-Sign"

    # Love / Rock / Horns: index and pinky extended (or thumb, index, pinky)
    if index > 0.7 and pinky > 0.7 and middle < 0.5 and ring < 0.5:
        if thumb > 0.7:
            return "ILY Sign"
        return "Rock Sign"

    # C-Shape / Claw: all fingers bent
    if 0.2 <= index <= 0.7 and 0.2 <= middle <= 0.7 and 0.2 <= ring <= 0.7:
        return "C-Shape"

    return "Partial Handshape"


def calculate_palm_orientation(landmarks: np.ndarray, is_left: bool = False) -> np.ndarray:
    """
    Calculate palm orientation features (9 dims):
    - Palm normal unit vector (3 dims) via cross product (Wrist -> Index MCP) x (Wrist -> Pinky MCP)
    - Palm pointing unit vector (3 dims) (Wrist -> Middle MCP)
    - Unnormalized wrist position (3 dims) for spatial tracking

    Returns:
        np.ndarray of shape (9,)
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(9, dtype=np.float32)

    lms = landmarks[:21, :3]
    wrist = lms[WRIST]
    index_mcp = lms[INDEX_MCP]
    pinky_mcp = lms[PINKY_MCP]
    middle_mcp = lms[MIDDLE_MCP]

    # Primary palm vectors
    v_index = index_mcp - wrist
    v_pinky = pinky_mcp - wrist

    # Normal vector: cross product
    if is_left:
        normal = np.cross(v_pinky, v_index)
    else:
        normal = np.cross(v_index, v_pinky)

    norm_mag = np.linalg.norm(normal)
    normal_unit = normal / (norm_mag + 1e-6)

    # Pointing vector
    v_middle = middle_mcp - wrist
    middle_mag = np.linalg.norm(v_middle)
    middle_unit = v_middle / (middle_mag + 1e-6)

    return np.concatenate([
        normal_unit.astype(np.float32),
        middle_unit.astype(np.float32),
        wrist.astype(np.float32),
    ])


def extract_single_hand_features(landmarks: Optional[np.ndarray], is_left: bool = False) -> np.ndarray:
    """
    Extract standardized 104-dimensional feature vector for a single hand:
    - 63 dims: Normalized 21 landmarks (x, y, z)
    - 15 dims: 15 finger joint angles
    - 12 dims: 12 scale-normalized finger distances
    -  5 dims: 5 finger extension states [Thumb, Index, Middle, Ring, Pinky]
    -  9 dims: Palm orientation (normal, pointing, wrist position)

    Total: 63 + 15 + 12 + 5 + 9 = 104 dims.
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    # 1. Normalized landmarks (63 dims)
    norm_lms, scale = normalize_hand_landmarks(landmarks)
    norm_flat = norm_lms.flatten()

    # 2. Joint angles (15 dims)
    angles = calculate_finger_joint_angles(landmarks)

    # 3. Finger distances (12 dims)
    distances = calculate_finger_distances(landmarks, scale=scale)

    # 4. Extension states (5 dims)
    states = classify_finger_states(landmarks)

    # 5. Palm orientation (9 dims)
    orientation = calculate_palm_orientation(landmarks, is_left=is_left)

    features = np.concatenate([
        norm_flat,       # 63
        angles,          # 15
        distances,       # 12
        states,          # 5
        orientation,     # 9
    ]).astype(np.float32)

    return features


def extract_two_hand_features(
    left_landmarks: Optional[np.ndarray] = None,
    right_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract unified 216-dimensional two-hand ISL feature vector:
    - Left hand features:  104 dims
    - Right hand features: 104 dims
    - Hand presence:         2 dims [is_left_present, is_right_present]
    - Relative hand spatial: 6 dims:
        * Inter-wrist displacement vector (3 dims): (right_wrist - left_wrist)
        * Inter-wrist distance (1 dim)
        * Min fingertip distance between hands (1 dim)
        * Palm normal alignment dot product (1 dim)

    Total: 104 + 104 + 2 + 6 = 216 dims.

    Guarantees strict left/right ordering and safe missing-hand handling.
    """
    has_left = left_landmarks is not None and len(left_landmarks) >= 21
    has_right = right_landmarks is not None and len(right_landmarks) >= 21

    # Extract single hand features
    left_feat = extract_single_hand_features(left_landmarks if has_left else None, is_left=True)
    right_feat = extract_single_hand_features(right_landmarks if has_right else None, is_left=False)

    # Presence flags
    presence = np.array([1.0 if has_left else 0.0, 1.0 if has_right else 0.0], dtype=np.float32)

    # Relative spatial features (6 dims)
    relative = np.zeros(6, dtype=np.float32)

    if has_left and has_right:
        lw = left_landmarks[WRIST][:3]
        rw = right_landmarks[WRIST][:3]

        # Displacement and distance
        displacement = rw - lw
        dist = np.linalg.norm(displacement)

        # Min fingertip distance
        left_tips = left_landmarks[FINGERTIP_INDICES, :3]
        right_tips = right_landmarks[FINGERTIP_INDICES, :3]
        # Pairwise distance matrix (5, 5)
        dists = np.linalg.norm(left_tips[:, None, :] - right_tips[None, :, :], axis=-1)
        min_tip_dist = np.min(dists)

        # Palm normal alignment dot product
        l_norm = left_feat[95:98]  # palm normal
        r_norm = right_feat[95:98]
        norm_dot = np.dot(l_norm, r_norm)

        relative[0:3] = displacement
        relative[3] = dist
        relative[4] = min_tip_dist
        relative[5] = norm_dot

    unified = np.concatenate([
        left_feat,   # 104
        right_feat,  # 104
        presence,    # 2
        relative,    # 6
    ]).astype(np.float32)

    return unified


def calculate_temporal_derivatives(sequence: np.ndarray) -> np.ndarray:
    """
    Calculate first-order velocity (diff) and second-order acceleration across sequence frames.

    Args:
        sequence: shape (T, D)

    Returns:
        derivatives: shape (T, D * 2) containing [velocity, acceleration]
    """
    if sequence is None or len(sequence) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Velocity: first difference
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])

    # Acceleration: second difference
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

    return np.concatenate([velocity, acceleration], axis=-1).astype(np.float32)
