"""Translation API endpoint."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "ta"


class TranslateResponse(BaseModel):
    translated_text: str
    source: str
    method: str
    error: Optional[str] = None


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Translate text between supported languages."""
    from backend.main import get_translation_service
    service = get_translation_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Translation service not initialised")
    
    result = service.translate(request.text, request.source_lang, request.target_lang)
    return TranslateResponse(**result)
