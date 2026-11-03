"""
Temporal filter: stabilizes gesture predictions over multiple frames.
Prevents flickering, blocks sign repetition while held,
and allows intentional repetition after a neutral pose.
"""

import logging
from collections import Counter, deque
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TemporalFilter:
    """
    Stabilizes predictions across multiple frames.

    Rules:
    - A gesture must appear `stability_frames` consecutive times to be accepted.
    - Once accepted, the same gesture is blocked for `repeat_block_frames`
      unless the buffer resets via a neutral/no-detection frame.
    - A None prediction for `neutral_frames` consecutive frames resets the block.
    """

    def __init__(
        self,
        stability_frames: int = 8,
        repeat_block_frames: int = 20,
        neutral_frames: int = 5,
    ):
        self._stability = stability_frames
        self._repeat_block = repeat_block_frames
        self._neutral_needed = neutral_frames

        self._recent: deque[Optional[str]] = deque(maxlen=stability_frames)
        self._last_accepted: Optional[str] = None
        self._frames_since_accepted: int = 0
        self._neutral_count: int = 0

    def update(self, prediction: Dict[str, Any]) -> Optional[str]:
        """
        Feed a new prediction dict and return a stable gesture if available.

        Args:
            prediction: dict with 'gesture' key (str or None) and 'status'.

        Returns:
            Accepted gesture string, or None.
        """
        gesture = prediction.get("gesture")
        status = prediction.get("status", "")

        # Count neutral frames for repeat-block reset
        if gesture is None:
            self._neutral_count += 1
            self._recent.append(None)
            if self._neutral_count >= self._neutral_needed:
                self._last_accepted = None
                self._frames_since_accepted = 0
            return None
        else:
            self._neutral_count = 0

        self._recent.append(gesture)
        self._frames_since_accepted += 1

        # Check stability: all recent frames must agree
        if len(self._recent) < self._stability:
            return None

        counts = Counter(self._recent)
        top_gesture, top_count = counts.most_common(1)[0]
        if top_count < self._stability or top_gesture is None:
            return None

        # Block repetition while sign is held
        if (
            top_gesture == self._last_accepted
            and self._frames_since_accepted < self._repeat_block
        ):
            return None

        # Accept
        self._last_accepted = top_gesture
        self._frames_since_accepted = 0
        self._neutral_count = 0
        return top_gesture

    def reset(self):
        """Fully reset the filter state."""
        self._recent.clear()
        self._last_accepted = None
        self._frames_since_accepted = 0
        self._neutral_count = 0
