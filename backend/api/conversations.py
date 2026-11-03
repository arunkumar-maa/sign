"""Conversation history API endpoints."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class SaveConversationRequest(BaseModel):
    gloss_sequence: List[str]
    english_text: str
    translated_text: str = ""
    language_code: str = "en"
    language_name: str = "English"


@router.post("/conversation")
async def save_conversation(request: SaveConversationRequest):
    """Save a conversation record to history."""
    from backend.main import get_database
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    
    record_id = db.save_conversation(
        gloss_sequence=request.gloss_sequence,
        english_text=request.english_text,
        translated_text=request.translated_text,
        language_code=request.language_code,
        language_name=request.language_name,
    )
    return {"id": record_id, "status": "saved"}


@router.get("/history")
async def get_history():
    """Get conversation history."""
    from backend.main import get_database
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    return {"records": db.load_history()}


@router.delete("/history")
async def clear_history():
    """Delete all history records."""
    from backend.main import get_database
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    count = db.delete_all()
    return {"deleted": count}


@router.delete("/history/{record_id}")
async def delete_record(record_id: str):
    """Delete a single history record."""
    from backend.main import get_database
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    deleted = db.delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"deleted": True}
