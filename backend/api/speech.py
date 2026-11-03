"""TTS and STT API endpoints."""

import logging
import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    lang_code: str = "en"


@router.post("/speech")
async def text_to_speech(request: TTSRequest):
    """Generate speech audio from text."""
    from backend.main import get_tts_service
    service = get_tts_service()
    if service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialised")
    
    path, error = service.synthesize(request.text, request.lang_code)
    if error:
        raise HTTPException(status_code=500, detail=error)
    return FileResponse(path, media_type="audio/mpeg", filename="speech.mp3")


@router.post("/speech-to-text")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    """Transcribe an uploaded audio file."""
    from backend.main import get_stt_service
    service = get_stt_service()
    if service is None:
        raise HTTPException(status_code=503, detail="STT service not initialised")

    # Save upload to temp file
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file too large (max 25 MB)")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = service.transcribe(tmp_path, language=language)
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
