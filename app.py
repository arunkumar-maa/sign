"""
SignBridge AI — Unified Application Entry Point.
Mounts the complete Gradio UI onto the FastAPI backend into a single ASGI application.

Usage:
    python app.py
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
import gradio as gr

import config as cfg
from backend.main import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("signbridge")


def build_gradio_app() -> gr.Blocks:
    """Build the complete Gradio Blocks application preserving all tabs and components."""
    from frontend.home import build_home_tab
    from frontend.diagnostics_ui import build_diagnostics_tab
    from frontend.translator_ui import build_translator_tab
    from isl_data_collection.ui import build_data_collection_tab
    from frontend.communication_ui import build_communication_tab
    from frontend.emergency_ui import build_emergency_tab
    from frontend.learning_ui import build_learning_tab
    from frontend.history_ui import build_history_tab
    from frontend.about_ui import build_about_tab

    with gr.Blocks(
        title="SignBridge AI — ISL Translator",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# 🤟 SignBridge AI &nbsp;&nbsp; *Breaking Communication Barriers with AI*"
        )

        build_home_tab()
        build_diagnostics_tab()
        build_translator_tab()
        build_data_collection_tab()
        build_communication_tab()
        build_emergency_tab()
        build_learning_tab()
        build_history_tab()
        build_about_tab()

        gr.Markdown(
            "---\n"
            "**Privacy:** Webcam frames are processed locally. "
            "No video is stored without your consent. "
            "TTS and translation require internet access. "
            "API docs: [/api/docs](/api/docs) &nbsp;|&nbsp; API info: [/api/info](/api/info)"
        )

    return demo


def create_unified_app():
    """
    Create and configure the unified FastAPI + Gradio ASGI application.
    Mounts Gradio Blocks at '/' and preserves FastAPI endpoints at '/api/*'.
    """
    fastapi_app = create_app()
    gradio_blocks = build_gradio_app()
    # Mount Gradio Blocks application at root '/'
    unified_app = gr.mount_gradio_app(fastapi_app, gradio_blocks, path="/")
    return unified_app


# Top-level ASGI application instance for Uvicorn
app = create_unified_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.API_PORT))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info("Starting SignBridge AI unified application on http://%s:%d...", host, port)
    logger.info("  - Homepage (Gradio UI): http://%s:%d/", host, port)
    logger.info("  - API Documentation:   http://%s:%d/api/docs", host, port)
    logger.info("  - API Health Check:    http://%s:%d/api/health", host, port)
    logger.info("  - API Information:     http://%s:%d/api/info", host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")
