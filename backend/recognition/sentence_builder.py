"""
SignBridge AI — Phase 6: Unified ISL Sentence Builder.
Maintains ordered sequence of confirmed signs/glosses in uppercase.
Strict stabilization and transition gating:
- Default mode: Manual confirmation (never auto-adds by default).
- Auto-add toggle: requires >= 15 consecutive frames (0.5s) and confidence >= 0.85.
- Neutral release gate: holding a sign never produces duplicate words.
- Rejection of unknown signs, stationary pseudo-words, and finger counts.
- Supports Accept Current Sign, Undo, Clear Sentence, and Reset.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Set

logger = logging.getLogger(__name__)

# Signs/labels that must never be added to a sentence
REJECTED_TOKENS: Set[str] = {
    "—", "-", "", "UNKNOWN", "UNKNOWN SIGN", "NONE", "NO HAND DETECTED",
    "COLLECTING SIGN FRAMES...", "COLLECTING SIGN FRAMES", "MODEL NOT TRAINED",
    "STATIONARY HAND", "STATIONARY", "HOLDING", "NEUTRAL", "WAITING FOR CAMERA...",
}


@dataclass
class WordEntry:
    gloss: str
    confidence: float
    timestamp: float


class SentenceBuilder:
    """
    Manages confirmed ISL signs and formats them into an ordered sentence.
    Includes strict stabilization and transition gating to eliminate duplicate words.
    """

    def __init__(
        self,
        gloss_to_english: Optional[Union[dict, float]] = None,
        repeat_cooldown_seconds: float = 1.5,
        required_stable_frames: int = 15,
        min_auto_confidence: float = 0.85,
    ):
        if isinstance(gloss_to_english, (int, float)):
            self.repeat_cooldown_seconds = float(gloss_to_english)
            self._gloss_map: Dict[str, str] = {}
        elif isinstance(gloss_to_english, dict):
            self._gloss_map = {k.upper(): v for k, v in gloss_to_english.items()}
            self.repeat_cooldown_seconds = repeat_cooldown_seconds
        else:
            self._gloss_map = {}
            self.repeat_cooldown_seconds = repeat_cooldown_seconds

        self.required_stable_frames = required_stable_frames
        self.min_auto_confidence = min_auto_confidence
        self.auto_add_enabled: bool = False  # Rule 1: Manual confirmation by default

        self._entries: List[WordEntry] = []
        self._user_override: Optional[str] = None
        self._last_added_time: float = 0.0

        # Stabilization tracking for auto-add
        self._candidate_sign: Optional[str] = None
        self._stable_frame_count: int = 0
        self._last_confirmed_sign: Optional[str] = None
        self._awaiting_neutral_release: bool = False

    def is_valid_sign(self, sign: Optional[str], min_confidence: float = 0.35) -> bool:
        """Check whether a sign candidate is valid and eligible for sentence insertion."""
        if not sign or not isinstance(sign, str):
            return False
        clean = sign.strip().upper()
        if clean in REJECTED_TOKENS:
            return False
        if clean.startswith("COUNT:") or clean.startswith("FINGER"):
            return False
        return True

    def add_word(self, word: Optional[str], confidence: float = 1.0) -> bool:
        """
        Add a confirmed word/sign to the sequence.
        Stores raw sign in uppercase (e.g. 'HELLO', 'WATER').
        """
        if not self.is_valid_sign(word):
            return False

        clean_gloss = word.strip().upper()
        # Clean prefix like "NUMBER: 2" -> "2"
        if clean_gloss.startswith("NUMBER:"):
            clean_gloss = clean_gloss.replace("NUMBER:", "").strip()

        now = time.perf_counter()

        # Cooldown duplicate prevention
        if self._entries:
            last = self._entries[-1].gloss
            if last == clean_gloss and (now - self._last_added_time) < self.repeat_cooldown_seconds:
                return False

        self._entries.append(WordEntry(gloss=clean_gloss, confidence=confidence, timestamp=now))
        self._last_added_time = now
        self._user_override = None
        self._last_confirmed_sign = clean_gloss
        self._awaiting_neutral_release = True
        logger.info("SentenceBuilder added gloss: '%s' (Total: %d)", clean_gloss, len(self._entries))
        return True

    def add_gloss(self, gesture: str, confidence: float = 1.0) -> bool:
        """Alias for add_word."""
        return self.add_word(gesture, confidence)

    def accept_current_sign(self, current_sign: Optional[str], confidence: float = 1.0) -> bool:
        """
        Explicit user action: 'Accept Current Sign' button.
        Bypasses auto-add stability count, but validates sign eligibility.
        """
        if not self.is_valid_sign(current_sign):
            return False
        return self.add_word(current_sign, confidence)

    def process_frame_auto_add(
        self,
        candidate_sign: Optional[str],
        confidence: float,
        hand_present: bool = True,
    ) -> bool:
        """
        Processes a streaming frame for auto-add with strict stabilization and neutral-release gating.
        Rule 3:
        - Hand must stabilize for at least 15 consecutive frames (0.5 s).
        - Prediction confidence must exceed 0.85.
        - Hand must return to neutral or change sign before the same sign can be added again.
        - Holding a sign in front of the camera must never produce duplicate words.
        """
        if not self.auto_add_enabled:
            return False

        # If hand leaves frame or returns to neutral, clear the release lock
        if not hand_present or not self.is_valid_sign(candidate_sign):
            self._candidate_sign = None
            self._stable_frame_count = 0
            self._awaiting_neutral_release = False
            return False

        clean = candidate_sign.strip().upper()
        if clean.startswith("NUMBER:"):
            clean = clean.replace("NUMBER:", "").strip()

        # If still holding the sign that was just added, do not add duplicate
        if self._awaiting_neutral_release:
            if clean == self._last_confirmed_sign:
                return False
            # Hand changed to a different valid sign -> clear lock
            self._awaiting_neutral_release = False
            self._candidate_sign = clean
            self._stable_frame_count = 1
            return False

        # Confidence gate (0.85)
        if confidence < self.min_auto_confidence:
            self._stable_frame_count = 0
            self._candidate_sign = None
            return False

        # Accumulate consecutive stable frames
        if clean == self._candidate_sign:
            self._stable_frame_count += 1
        else:
            self._candidate_sign = clean
            self._stable_frame_count = 1

        # Check threshold
        if self._stable_frame_count >= self.required_stable_frames:
            added = self.add_word(clean, confidence)
            if added:
                self._candidate_sign = None
                self._stable_frame_count = 0
                return True

        return False

    def undo(self) -> Optional[str]:
        """Remove the most recently added word (Undo Last Word button)."""
        if self._entries:
            removed = self._entries.pop()
            self._user_override = None
            self._awaiting_neutral_release = False
            word_str = self._gloss_map.get(removed.gloss, removed.gloss.capitalize())
            logger.info("SentenceBuilder undo: removed '%s'", word_str)
            return word_str
        return None

    def clear(self):
        """Clear all words and reset sentence (Clear Sentence button)."""
        self._entries.clear()
        self._user_override = None
        self._last_added_time = 0.0
        self._candidate_sign = None
        self._stable_frame_count = 0
        self._last_confirmed_sign = None
        self._awaiting_neutral_release = False
        logger.info("SentenceBuilder cleared.")

    def reset(self):
        """Full reset alias."""
        self.clear()

    def set_user_text(self, text: str):
        """Set manual user-edited sentence text."""
        self._user_override = text

    @property
    def words(self) -> List[str]:
        """Ordered list of recognized words (formatted/capitalized)."""
        return [self._gloss_map.get(e.gloss, e.gloss.capitalize()) for e in self._entries]

    @property
    def gloss_sequence(self) -> List[str]:
        """Ordered list of recognized glosses in uppercase."""
        return [e.gloss for e in self._entries]

    @property
    def current_sentence(self) -> str:
        """Current formatted sentence string."""
        if self._user_override is not None:
            return self._user_override

        if not self._entries:
            return ""

        parts = []
        for e in self._entries:
            mapped = self._gloss_map.get(e.gloss, e.gloss.capitalize())
            parts.append(mapped)

        sentence = " ".join(parts)
        if sentence and not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    @property
    def suggested_sentence(self) -> str:
        """Auto-generated sentence string."""
        return self.current_sentence

    @property
    def confirmed_text(self) -> str:
        """Return user-edited text if set, else current sentence."""
        if self._user_override is not None:
            return self._user_override
        return self.current_sentence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "words": self.words,
            "gloss_sequence": self.gloss_sequence,
            "current_sentence": self.current_sentence,
            "suggested_sentence": self.suggested_sentence,
            "confirmed_text": self.confirmed_text,
            "word_count": len(self._entries),
            "is_user_edited": self._user_override is not None,
            "auto_add_enabled": self.auto_add_enabled,
        }
