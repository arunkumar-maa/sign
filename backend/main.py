"""
SignBridge AI — FastAPI application factory.
Services are initialised once and stored as module-level singletons.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config as cfg

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_static_recognizer = None
_dynamic_recognizer = None
_translation_service = None
_tts_service = None
_stt_service = None
_database = None
_animation_mapper = None


def get_static_recognizer():
    return _static_recognizer

def get_dynamic_recognizer():
    return _dynamic_recognizer

def get_translation_service():
    return _translation_service

def get_tts_service():
    return _tts_service

def get_stt_service():
    return _stt_service

def get_database():
    return _database

def get_animation_mapper():
    return _animation_mapper


def _init_services():
    """Initialise all backend services at startup."""
    global _static_recognizer, _dynamic_recognizer, _translation_service
    global _tts_service, _stt_service, _database, _animation_mapper

    # Static recognizer
    try:
        from backend.recognition.static_recognizer import StaticRecognizer
        _static_recognizer = StaticRecognizer(
            model_path=cfg.STATIC_MODEL_PATH,
            labels_path=cfg.STATIC_LABELS_PATH,
            confidence_threshold=cfg.STATIC_CONFIDENCE_THRESHOLD,
        )
    except Exception as exc:
        logger.error("Failed to init static recognizer: %s", exc)

    # Dynamic recognizer
    try:
        from backend.recognition.dynamic_recognizer import DynamicRecognizer
        _dynamic_recognizer = DynamicRecognizer(
            model_path=cfg.DYNAMIC_MODEL_PATH,
            labels_path=cfg.DYNAMIC_LABELS_PATH,
            sequence_length=cfg.DYNAMIC_SEQUENCE_LENGTH,
            confidence_threshold=cfg.DYNAMIC_CONFIDENCE_THRESHOLD,
        )
    except Exception as exc:
        logger.error("Failed to init dynamic recognizer: %s", exc)

    # Translation service
    try:
        from backend.translation.translator import TranslationService
        _translation_service = TranslationService(
            libre_url=cfg.LIBRE_TRANSLATE_URL,
            libre_api_key=cfg.LIBRE_TRANSLATE_API_KEY,
            offline_dict=cfg.ISL_PHRASE_TRANSLATIONS,
        )
    except Exception as exc:
        logger.error("Failed to init translation service: %s", exc)

    # TTS service
    try:
        from backend.audio.text_to_speech import TTSService
        _tts_service = TTSService(
            provider=cfg.TTS_PROVIDER,
            voice_map=cfg.EDGE_TTS_VOICES,
            gtts_lang_map=cfg.GTTS_LANG_MAP,
        )
    except Exception as exc:
        logger.error("Failed to init TTS service: %s", exc)

    # STT service
    try:
        from backend.audio.speech_to_text import SpeechToTextService
        _stt_service = SpeechToTextService(
            whisper_model_size=cfg.WHISPER_MODEL_SIZE,
            whisper_device=cfg.WHISPER_DEVICE,
            whisper_compute_type=cfg.WHISPER_COMPUTE_TYPE,
            tlive_enabled=cfg.TLIVE_ENABLED,
            tlive_base_url=cfg.TLIVE_BASE_URL,
            tlive_model=cfg.TLIVE_MODEL,
            tlive_api_key=cfg.TLIVE_API_KEY,
        )
    except Exception as exc:
        logger.error("Failed to init STT service: %s", exc)

    # Database
    try:
        from backend.database.database import Database
        _database = Database(db_path=cfg.DB_PATH)
    except Exception as exc:
        logger.error("Failed to init database: %s", exc)

    # Animation mapper
    try:
        from backend.avatar.animation_mapping import AnimationMapper
        _animation_mapper = AnimationMapper(manifest_path=cfg.ANIMATION_MANIFEST_PATH)
    except Exception as exc:
        logger.error("Failed to init animation mapper: %s", exc)

    logger.info("All services initialised.")


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI Lifespan context manager for startup and shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _init_services()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SignBridge AI",
        description="Indian Sign Language Alphabet and Numbers Recognition",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    allowed_origins = getattr(cfg, "ALLOWED_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins else ["*"],
        allow_credentials=True if allowed_origins and "*" not in allowed_origins else False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from backend.api.health import router as health_router
    from backend.api.recognition import router as recognition_router
    from backend.api.translation import router as translation_router
    from backend.api.speech import router as speech_router
    from backend.api.conversations import router as conversations_router

    prefix = "/api"
    app.include_router(health_router, prefix=prefix, tags=["Health"])
    app.include_router(recognition_router, prefix=prefix, tags=["Recognition"])
    app.include_router(translation_router, prefix=prefix, tags=["Translation"])
    app.include_router(speech_router, prefix=prefix, tags=["Audio"])
    app.include_router(conversations_router, prefix=prefix, tags=["History"])

    return app


# Module-level app instance (for uvicorn)
app = create_app()
