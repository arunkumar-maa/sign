import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    MIN_RECORDING_FRAMES,
    MAX_RECORDING_SECONDS,
    RECORDING_FPS,
    VIDEO_DIR,
)
from .storage import append_sample, clean_token, utc_now, validate_label


@dataclass
class ActiveRecording:
    sample_id: str
    category: str
    label: str
    signer_id: str
    path: Path
    started_at: float
    writer: Optional[cv2.VideoWriter] = None
    frame_count: int = 0
    hands_detected_count: int = 0
    width: int = 0
    height: int = 0
    auto_stopped: bool = False


class RecordingManager:
    """Stores clean webcam frames (without landmark overlays) in labelled clips."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveRecording] = {}
        self._lock = threading.RLock()

    def start(self, session_id: str, category: str, label: str, signer_id: str, consent: bool) -> str:
        if not consent:
            raise ValueError("Confirm that the signer has given permission before recording.")
        if not session_id:
            raise ValueError("Browser session is missing. Refresh the page and try again.")
        category, label = validate_label(category, label)
        signer_id = clean_token(signer_id, "signer ID", max_length=32)
        with self._lock:
            if session_id in self._active:
                raise ValueError("A recording is already active. Stop it before starting another.")
            sample_id = uuid.uuid4().hex[:12]
            filename = f"{category}_{label}_{signer_id}_{sample_id}.mp4"
            self._active[session_id] = ActiveRecording(
                sample_id=sample_id,
                category=category,
                label=label,
                signer_id=signer_id,
                path=VIDEO_DIR / filename,
                started_at=time.monotonic(),
            )
        return f"Recording started: {category.upper()} / {label} / signer {signer_id}. Show one complete sign, then click Stop Recording."

    def is_active(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._active

    def append_frame(self, session_id: str, rgb_frame: np.ndarray, has_hands: bool = True) -> str:
        if not isinstance(rgb_frame, np.ndarray) or rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
            return "Camera frame is invalid."
        with self._lock:
            recording = self._active.get(session_id)
            if recording is None:
                return "Camera ready. Select a label and click Start Recording."

            height, width = rgb_frame.shape[:2]
            if recording.writer is None:
                recording.width, recording.height = width, height
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                recording.writer = cv2.VideoWriter(
                    str(recording.path), fourcc, RECORDING_FPS, (width, height)
                )
                if not recording.writer.isOpened():
                    recording.writer.release()
                    recording.writer = None
                    self._active.pop(session_id, None)
                    raise RuntimeError("Could not create MP4 recording. Check OpenCV video codec support.")

            frame = rgb_frame[:, :, :3]
            if frame.shape[1] != recording.width or frame.shape[0] != recording.height:
                frame = cv2.resize(frame, (recording.width, recording.height))
            recording.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            recording.frame_count += 1
            if has_hands:
                recording.hands_detected_count += 1

            elapsed = time.monotonic() - recording.started_at
            if elapsed >= MAX_RECORDING_SECONDS:
                recording.auto_stopped = True
                self._finish_locked(session_id)
                return f"Maximum {MAX_RECORDING_SECONDS:.0f}-second clip saved. Start another sample for this label."

            return f"Recording {recording.label}: {recording.frame_count} frames ({recording.hands_detected_count} hand frames), {elapsed:.1f}/{MAX_RECORDING_SECONDS:.0f}s."

    def stop(self, session_id: str) -> str:
        with self._lock:
            if session_id not in self._active:
                return "No recording is active."
            return self._finish_locked(session_id)

    def _finish_locked(self, session_id: str) -> str:
        recording = self._active.pop(session_id)
        if recording.writer is not None:
            recording.writer.release()
        if recording.frame_count < MIN_RECORDING_FRAMES:
            recording.path.unlink(missing_ok=True)
            return (
                f"Sample discarded: only {recording.frame_count} frames were captured. "
                f"Record at least {MIN_RECORDING_FRAMES} frames."
            )
        if recording.hands_detected_count < 3:
            recording.path.unlink(missing_ok=True)
            return (
                f"Sample discarded: no valid hand landmarks detected ({recording.hands_detected_count} hand frames). "
                f"Please ensure hand is clearly visible to camera."
            )
        if not recording.path.exists() or recording.path.stat().st_size == 0:
            return "Recording file was not created. Check OpenCV video codec support."

        append_sample({
            "sample_id": recording.sample_id,
            "category": recording.category,
            "label": recording.label,
            "signer_id": recording.signer_id,
            "video_path": str(recording.path.relative_to(VIDEO_DIR.parent.parent)),
            "created_at": utc_now(),
            "fps": RECORDING_FPS,
            "width": recording.width,
            "height": recording.height,
            "frame_count": recording.frame_count,
            "consent_confirmed": True,
            "source": "self_recorded",
            "split_policy": "signer_independent",
        })
        return (
            f"Saved {recording.category.upper()} sample: {recording.label}; "
            f"signer {recording.signer_id}; {recording.frame_count} frames; "
            f"{recording.path.name}"
        )


recording_manager = RecordingManager()
