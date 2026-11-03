"""Health check API endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
import time
import platform
import sys

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: float
    python_version: str
    platform: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns service health status."""
    return HealthResponse(
        status="ok",
        service="SignBridge AI",
        version="0.1.0",
        timestamp=time.time(),
        python_version=sys.version,
        platform=platform.platform(),
    )
