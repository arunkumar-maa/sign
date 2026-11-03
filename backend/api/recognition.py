"""Recognition API endpoint."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


class RecognizeRequest(BaseModel):
    features: List[float] = Field(..., description="Flattened landmark feature vector")
    recognition_type: str = Field(default="static", description="'static' or 'dynamic'")


class RecognizeResponse(BaseModel):
    gesture: Optional[str]
    confidence: float
    status: str
    recognition_type: str
    model_version: str
    timestamp: float
    message: Optional[str] = None


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_gesture(request: RecognizeRequest):
    """
    Run gesture recognition on a feature vector.
    The recognizer is injected at startup via app state.
    """
    from backend.main import get_static_recognizer, get_dynamic_recognizer

    try:
        features = np.array(request.features, dtype=np.float32)

        if request.recognition_type == "dynamic":
            recognizer = get_dynamic_recognizer()
            if recognizer is None:
                raise HTTPException(status_code=503, detail="Dynamic recognizer not initialised")
            recognizer.add_frame(features)
            result = recognizer.predict()
        else:
            recognizer = get_static_recognizer()
            if recognizer is None:
                raise HTTPException(status_code=503, detail="Static recognizer not initialised")
            result = recognizer.predict(features)

        return RecognizeResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Recognition API error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
