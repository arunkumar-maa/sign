"""
SignBridge AI — Character Text Builder for ISL Alphabet & Numbers.
Assembles recognized characters (A–Z and 0–9) into readable text (e.g. "HELLO123").

Features:
- Character-by-character text generation (not word glosses).
- Explicit controls: Add Character, Backspace, Clear Text.
- Auto-add mode with stability gating:
  - Sign must be stably detected for N consecutive frames.
  - Requires neutral release (no hand or neutral pose) before repeating the same character.
  - Prevents rapid runaway additions (e.g. prevents "AAAAA").
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

INVALID_CHARS: Set[str] = {
    "—", "-", "", "UNKNOWN", "UNKNOWN SIGN", "NONE", "NO HAND DETECTED",
    "COLLECTING", "MODEL NOT TRAINED", "MODEL NOT AVAILABLE",
    "NEUTRAL", "WAITING FOR CAMERA", "?",
}

VALID_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VALID_NUMBERS = set("0123456789")
VALID_CHARACTERS = VALID_ALPHABET | VALID_NUMBERS


@dataclass
class CharacterEntry:
    character: str
    confidence: float
    timestamp: float


class CharacterTextBuilder:
    """
    Manages character accumulation into readable strings (e.g. 'HELLO123').
    Designed for ISL Alphabet (A-Z) and Number (0-9) recognition.
    """

    def __init__(
        self,
        required_stable_frames: int = 6,
        min_auto_confidence: float = 0.65,
        repeat_cooldown_seconds: float = 1.0,
    ):
        self.required_stable_frames = required_stable_frames
        self.min_auto_confidence = min_auto_confidence
        self.repeat_cooldown_seconds = repeat_cooldown_seconds
        self.auto_add_enabled: bool = False

        self._text: str = ""
        self._history: List[CharacterEntry] = []
        self._last_added_time: float = 0.0

        # Auto-add stabilization state
        self._candidate_char: Optional[str] = None
        self._stable_count: int = 0
        self._awaiting_neutral_release: bool = False
        self._neutral_frame_count: int = 0

    @property
    def text(self) -> str:
        return self._text

    @property
    def current_sentence(self) -> str:
        """Alias for compatibility with existing UI components."""
        return self._text

    @property
    def confirmed_text(self) -> str:
        return self._text

    @property
    def history(self) -> List[CharacterEntry]:
        return list(self._history)

    def is_valid_char(self, char_candidate: Optional[str]) -> bool:
        if not char_candidate or not isinstance(char_candidate, str):
            return False
        clean = char_candidate.strip().upper()
        if len(clean) != 1:
            # Handle potential strings like "LETTER A" or "NUMBER: 5"
            parts = clean.split()
            if parts and len(parts[-1]) == 1:
                clean = parts[-1]
            else:
                return False
        return clean in VALID_CHARACTERS

    def clean_char(self, char_candidate: str) -> Optional[str]:
        clean = char_candidate.strip().upper()
        if len(clean) == 1 and clean in VALID_CHARACTERS:
            return clean
        parts = clean.split()
        if parts and len(parts[-1]) == 1 and parts[-1] in VALID_CHARACTERS:
            return parts[-1]
        return None

    def add_character(self, character: Optional[str], confidence: float = 1.0) -> bool:
        """
        Manually or programmatically append a character to the text.
        Returns True if character was appended, False otherwise.
        """
        if not character:
            return False
        c = self.clean_char(character)
        if not c:
            return False

        now = time.perf_counter()
        self._text += c
        self._history.append(CharacterEntry(character=c, confidence=confidence, timestamp=now))
        self._last_added_time = now
        logger.info("CharacterTextBuilder: Added '%s' -> Current text: '%s'", c, self._text)
        return True

    def backspace(self) -> str:
        """Delete the last character from the accumulated text."""
        if self._text:
            deleted = self._text[-1]
            self._text = self._text[:-1]
            if self._history:
                self._history.pop()
            logger.info("CharacterTextBuilder: Backspace removed '%s' -> Current text: '%s'", deleted, self._text)
        # Reset auto-add gating on explicit edit
        self._candidate_char = None
        self._stable_count = 0
        self._awaiting_neutral_release = False
        return self._text

    def clear(self) -> str:
        """Clear the accumulated text and reset all tracking."""
        self._text = ""
        self._history.clear()
        self._candidate_char = None
        self._stable_count = 0
        self._awaiting_neutral_release = False
        self._neutral_frame_count = 0
        logger.info("CharacterTextBuilder: Cleared text.")
        return ""

    def set_text(self, text: str) -> str:
        """Directly set or edit the text buffer."""
        self._text = str(text) if text is not None else ""
        return self._text

    def process_auto_frame(
        self,
        character_candidate: Optional[str],
        confidence: float,
        hand_detected: bool,
    ) -> Optional[str]:
        """
        Process a frame for auto-add stabilization.
        Returns the newly added character if one was added on this frame, else None.
        """
        if not self.auto_add_enabled:
            return None

        # If no hand detected or invalid sign, increment neutral frame counter
        if not hand_detected or not self.is_valid_char(character_candidate):
            self._neutral_frame_count += 1
            if self._neutral_frame_count >= 3:
                # User has released the hand / reset to neutral
                self._awaiting_neutral_release = False
                self._candidate_char = None
                self._stable_count = 0
            return None

        c = self.clean_char(character_candidate)
        if not c or confidence < self.min_auto_confidence:
            self._neutral_frame_count += 1
            if self._neutral_frame_count >= 3:
                self._awaiting_neutral_release = False
                self._candidate_char = None
                self._stable_count = 0
            return None

        # Valid candidate sign detected
        self._neutral_frame_count = 0

        # Check if user is holding the same character that was already added
        if self._awaiting_neutral_release and self._candidate_char == c:
            # Must release hand before repeating the same character
            return None

        if self._candidate_char == c:
            self._stable_count += 1
        else:
            self._candidate_char = c
            self._stable_count = 1

        if self._stable_count >= self.required_stable_frames:
            # Confirm and append
            self.add_character(c, confidence=confidence)
            self._awaiting_neutral_release = True
            self._stable_count = 0
            return c

        return None

    # Compatibility aliases for SentenceBuilder callers
    @property
    def words(self) -> List[str]:
        return [entry.character for entry in self._history]

    @property
    def gloss_sequence(self) -> List[str]:
        return [entry.character for entry in self._history]

    def add_word(self, word: Optional[str], confidence: float = 1.0) -> bool:
        if not word:
            return False
        clean = str(word).strip()
        if not clean:
            return False
        # If single char
        if len(clean) == 1 and clean.upper() in VALID_CHARACTERS:
            return self.add_character(clean.upper(), confidence=confidence)
        # Multi-char word (e.g. from tests or glosses)
        now = time.perf_counter()
        token = clean.capitalize()
        if self._text:
            self._text += " " + token
        else:
            self._text = token
        self._history.append(CharacterEntry(character=token, confidence=confidence, timestamp=now))
        self._last_added_time = now
        return True

    def accept_current_sign(self, sign: Optional[str], confidence: float = 1.0) -> bool:
        return self.add_character(sign, confidence=confidence)

    def undo(self) -> str:
        return self.backspace()

    def clear_all(self) -> str:
        return self.clear()

    def set_user_text(self, text: str) -> str:
        return self.set_text(text)

    def process_frame_auto_add(
        self,
        candidate_sign: Optional[str],
        confidence: float,
        hand_present: bool = True,
    ) -> Optional[str]:
        return self.process_auto_frame(
            character_candidate=candidate_sign,
            confidence=confidence,
            hand_detected=hand_present,
        )
