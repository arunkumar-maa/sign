"""
Speech-to-text using Faster-Whisper (primary).
Optional adapter for a separately running TLive-Omni server.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_TEMP_DIR = Path(tempfile.gettempdir()) / "signbridge_stt"
_TEMP_DIR.mkdir(exist_ok=True)

_WHISPER_MODEL = None  # lazy-loaded


def _get_whisper_model(model_size: str, device: str, compute_type: str):
    """Lazy-load Faster-Whisper model (singleton)."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading Faster-Whisper model '%s' on %s...", model_size, device)
            _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Faster-Whisper model loaded.")
        except Exception as exc:
            logger.error("Failed to load Faster-Whisper: %s", exc)
    return _WHISPER_MODEL


class SpeechToTextService:
    """
    Transcribes audio to text.
    Primary: Faster-Whisper.
    Optional: TLive-Omni adapter (separate server required).
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        tlive_enabled: bool = False,
        tlive_base_url: str = "",
        tlive_model: str = "",
        tlive_api_key: str = "",
    ):
        self._ws_size = whisper_model_size
        self._ws_device = whisper_device
        self._ws_compute = whisper_compute_type
        self._tlive_enabled = tlive_enabled
        self._tlive_url = tlive_base_url.rstrip("/")
        self._tlive_model = tlive_model
        self._tlive_key = tlive_api_key
        logger.info("SpeechToTextService ready (whisper=%s, tlive=%s)", whisper_model_size, tlive_enabled)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: BCP-47 language code or None for auto-detect.
            task: 'transcribe' or 'translate'.

        Returns:
            dict with 'text', 'language', 'method', and optional 'error'.
        """
        path = Path(audio_path)
        if not path.exists():
            return {"text": "", "language": language, "method": "error", "error": "Audio file not found."}

        # Validate file size (max 25 MB)
        if path.stat().st_size > 25 * 1024 * 1024:
            return {"text": "", "language": language, "method": "error", "error": "Audio file too large (max 25 MB)."}

        # Try TLive-Omni adapter if enabled
        if self._tlive_enabled:
            result = self._transcribe_tlive(path, language, task)
            if result.get("method") != "error":
                return result

        # Primary: Faster-Whisper
        return self._transcribe_whisper(path, language, task)

    def _transcribe_whisper(self, path: Path, language: Optional[str], task: str) -> dict:
        model = _get_whisper_model(self._ws_size, self._ws_device, self._ws_compute)
        if model is None:
            return {
                "text": "",
                "language": language,
                "method": "error",
                "error": "Faster-Whisper model failed to load. Check that faster-whisper is installed.",
            }
        try:
            segments, info = model.transcribe(
                str(path),
                language=language,
                task=task,
                beam_size=5,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return {
                "text": text,
                "language": info.language,
                "method": "faster-whisper",
            }
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return {"text": "", "language": language, "method": "error", "error": str(exc)}

    def _transcribe_tlive(self, path: Path, language: Optional[str], task: str) -> dict:
        """
        Optional adapter: calls a separately running TLive-Omni vLLM server
        at POST /v1/audio/transcriptions (OpenAI-compatible endpoint).
        
        This adapter follows the API structure studied from:
        TaoLiveAIGC/TLive-Omni/vllm/vllm/entrypoints/openai/speech_to_text/
        Note: TLive-Omni's `to_language` field is currently a placeholder.
        """
        if not self._tlive_url:
            return {"text": "", "language": language, "method": "error", "error": "TLive URL not configured."}
        try:
            headers = {}
            if self._tlive_key:
                headers["Authorization"] = f"Bearer {self._tlive_key}"

            with open(path, "rb") as f:
                files = {"file": (path.name, f, "audio/mpeg")}
                data = {"model": self._tlive_model or "default"}
                if language:
                    data["language"] = language
                if task == "translate":
                    data["task"] = "translate"

                resp = requests.post(
                    f"{self._tlive_url}/v1/audio/transcriptions",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60,
                )
            resp.raise_for_status()
            result = resp.json()
            return {
                "text": result.get("text", ""),
                "language": result.get("language", language),
                "method": "tlive-omni",
            }
        except Exception as exc:
            logger.warning("TLive-Omni adapter failed: %s", exc)
            return {"text": "", "language": language, "method": "error", "error": str(exc)}
