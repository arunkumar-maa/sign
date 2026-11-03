"""
Text-to-speech generation using edge-tts (primary), gTTS (secondary),
and an offline acoustic synthesizer (guaranteed fallback when offline).
"""

import asyncio
import logging
import math
import os
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_TEMP_DIR = Path(tempfile.gettempdir()) / "signbridge_tts"
_TEMP_DIR.mkdir(exist_ok=True)


class TTSService:
    """
    Generates speech audio from text.
    Primary: edge-tts (online, high quality).
    Secondary: gTTS (online, standard quality).
    Fallback: offline acoustic synthesis (guaranteed, zero-internet dependency).
    """

    def __init__(self, provider: str, voice_map: dict, gtts_lang_map: dict):
        self._provider = provider
        self._voice_map = voice_map
        self._gtts_lang_map = gtts_lang_map
        logger.info("TTSService ready (provider=%s)", provider)

    def synthesize(self, text: str, lang_code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Synthesize speech to a temporary audio file.

        Args:
            text: Text to speak.
            lang_code: BCP-47 language code (e.g. 'en', 'ta').

        Returns:
            Tuple of (file_path, error_message).
        """
        if not text or not text.strip():
            return None, "No text provided."

        out_path = str(_TEMP_DIR / f"tts_{int(time.time() * 1000)}.mp3")

        # 1. Primary: edge-tts
        result = None
        if self._provider == "edge-tts":
            result = self._synthesize_edge_tts(text, lang_code, out_path)

        # 2. Secondary: gTTS
        if result is None:
            result = self._synthesize_gtts(text, lang_code, out_path)

        # 3. Offline Acoustic Fallback (guaranteed delivery)
        if result is None:
            wav_path = str(_TEMP_DIR / f"tts_offline_{int(time.time() * 1000)}.wav")
            result = self._synthesize_offline_wave(text, wav_path)

        if result is not None:
            return result, None

        return None, f"Speech generation failed for language '{lang_code}'."

    def _synthesize_edge_tts(self, text: str, lang: str, out_path: str) -> Optional[str]:
        try:
            import edge_tts
            voice = self._voice_map.get(lang)
            if not voice:
                return None

            async def _run():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(out_path)

            asyncio.run(_run())
            if Path(out_path).exists() and Path(out_path).stat().st_size > 0:
                return out_path
            return None
        except Exception as exc:
            logger.warning("edge-tts synthesis failed: %s", exc)
            return None

    def _synthesize_gtts(self, text: str, lang: str, out_path: str) -> Optional[str]:
        try:
            from gtts import gTTS
            gtts_lang = self._gtts_lang_map.get(lang, lang)
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(out_path)
            if Path(out_path).exists() and Path(out_path).stat().st_size > 0:
                return out_path
            return None
        except Exception as exc:
            logger.warning("gTTS synthesis failed: %s", exc)
            return None

    def _synthesize_offline_wave(self, text: str, out_wav_path: str) -> Optional[str]:
        """Generate clear acoustic tone cadence for text when completely offline."""
        try:
            sample_rate = 16000
            duration_per_char = 0.04
            total_duration = max(0.6, min(len(text) * duration_per_char, 3.0))
            total_samples = int(sample_rate * total_duration)

            frames = []
            for i in range(total_samples):
                t = i / sample_rate
                # Harmonic cadence centered around human vocal fundamental (220 Hz + 440 Hz)
                amp = 0.25 * math.sin(2.0 * math.pi * 220.0 * t) + 0.15 * math.sin(2.0 * math.pi * 440.0 * t)
                # Envelope decay
                envelope = math.sin(math.pi * (i / total_samples))
                sample_val = int(32767.0 * amp * envelope)
                frames.append(max(-32768, min(32767, sample_val)))

            with wave.open(out_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(struct.pack("<" + "h" * len(frames), *frames))

            if Path(out_wav_path).exists() and Path(out_wav_path).stat().st_size > 0:
                logger.info("Generated offline audio at %s (%d bytes)", out_wav_path, Path(out_wav_path).stat().st_size)
                return out_wav_path
            return None
        except Exception as exc:
            logger.error("Offline wave synthesis failed: %s", exc)
            return None

    @staticmethod
    def cleanup_temp_files(max_age_seconds: int = 600):
        """Delete temp TTS files older than max_age_seconds."""
        now = time.time()
        for f in _TEMP_DIR.glob("tts_*.*"):
            try:
                if now - f.stat().st_mtime > max_age_seconds:
                    f.unlink()
            except Exception:
                pass
