"""
Ayzek Salimli Sign Language Recognizer Integration.
Direct geometric landmark classification based on ayzeksalimli/sign-language-to-text.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Add ayzek repository to path
AYZEK_PATH = Path(__file__).parent.parent.parent.parent / "references" / "ayzek_slt"
if str(AYZEK_PATH) not in sys.path:
    sys.path.insert(0, str(AYZEK_PATH))

try:
    from app.languages import ASL
    from app.alphabet.spec import best_matches, LetterSpec, EXT, FOLD, HALF, T_OUT, T_ACROSS, T_UP
    from app.recognition.geometry import extract
    _AYZEK_AVAILABLE = True
except Exception as e:
    logger.warning("Could not import ayzek_slt engine: %s", e)
    _AYZEK_AVAILABLE = False


class AyzekRecognizer:
    """
    Direct geometric sign language classifier powered by ayzeksalimli/sign-language-to-text.
    """

    def __init__(self, min_score: float = 0.65):
        self.min_score = min_score
        self.specs = []
        if _AYZEK_AVAILABLE:
            self.specs = list(ASL.specs)

    def predict(self, landmarks_flat: np.ndarray) -> Dict[str, Any]:
        """
        Predict sign from flat landmarks.
        landmarks_flat contains at least 63 floats (21 x,y,z points for the hand).
        """
        if not _AYZEK_AVAILABLE or not self.specs:
            return {"gesture": None, "confidence": 0.0, "status": "unavailable"}

        try:
            arr = np.array(landmarks_flat, dtype=np.float32).flatten()
            if len(arr) < 63:
                return {"gesture": None, "confidence": 0.0, "status": "no_hands"}

            # Extract first 63 values (hand)
            hand_pts = arr[:63].reshape(21, 3)
            # If all zeros, no hand
            if np.all(np.abs(hand_pts) < 1e-4):
                return {"gesture": None, "confidence": 0.0, "status": "no_hands"}

            pts: List[Tuple[float, float, float]] = [
                (float(p[0]), float(p[1]), float(p[2])) for p in hand_pts
            ]

            feats = extract(pts, "Right")

            # Check secondary hand if present (index 63..125)
            has_two_hands = len(arr) >= 126 and not np.all(np.abs(arr[63:126]) < 1e-4)

            if has_two_hands:
                sec_pts = [(float(p[0]), float(p[1]), float(p[2])) for p in arr[63:126].reshape(21, 3)]
                sec_feats = extract(sec_pts, "Left")
                # HELP: One fist on flat open hand
                if (feats.is_fist() and sec_feats.is_open_palm()) or (sec_feats.is_fist() and feats.is_open_palm()):
                    return {
                        "gesture": "HELP",
                        "confidence": 0.96,
                        "status": "recognized",
                        "engine": "ayzek_geometric",
                    }
                # EMERGENCY: Both hands open and shaking
                if feats.is_open_palm() and sec_feats.is_open_palm():
                    return {
                        "gesture": "EMERGENCY",
                        "confidence": 0.94,
                        "status": "recognized",
                        "engine": "ayzek_geometric",
                    }

            # Check macro signs
            # HELLO: Open 5 fingers wide spread
            if feats.is_open_palm():
                return {
                    "gesture": "HELLO",
                    "confidence": 0.95,
                    "status": "recognized",
                    "engine": "ayzek_geometric",
                }

            # WATER: W handshape (index, middle, ring extended, thumb holding pinky)
            if (feats.extended.get("index") and feats.extended.get("middle") and 
                feats.extended.get("ring") and feats.folded.get("pinky")):
                return {
                    "gesture": "WATER",
                    "confidence": 0.95,
                    "status": "recognized",
                    "engine": "ayzek_geometric",
                }

            # Run alphabet matching
            matches = best_matches(feats, self.specs, top=2)
            if matches:
                top = matches[0]
                if top.score >= self.min_score:
                    return {
                        "gesture": top.letter,
                        "confidence": float(top.score),
                        "status": "recognized",
                        "engine": "ayzek_geometric",
                    }

            # Fallback to general fist / nod
            if feats.is_fist():
                return {
                    "gesture": "A",
                    "confidence": 0.85,
                    "status": "recognized",
                    "engine": "ayzek_geometric",
                }

            return {"gesture": None, "confidence": 0.0, "status": "no_match"}
        except Exception as e:
            logger.error("Ayzek prediction error: %s", e)
            return {"gesture": None, "confidence": 0.0, "status": "error", "error": str(e)}
