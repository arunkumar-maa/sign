"""
SignBridge AI — Main application entry point.
Starts FastAPI backend and Gradio frontend in a single process.

Usage:
    .venv/Scripts/python.exe app.py
"""

import logging
import sys
import threading
from pathlib import Path

# Add project root to Python path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
import gradio as gr

import config as cfg
from backend.main import app as fastapi_app, _init_services

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("signbridge")

# Initialise backend services immediately (not waiting for uvicorn startup event)
_init_services()


def build_gradio_app() -> gr.Blocks:
    """Build the complete Gradio Blocks application."""
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
            "API docs: [/api/docs](/api/docs)"
        )

    return demo


def start_fastapi():
    """Run FastAPI in a background thread."""
    uvicorn.run(
        fastapi_app,
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    logger.info("Starting SignBridge AI...")
    logger.info("FastAPI will be available at http://%s:%d/api/docs", cfg.API_HOST, cfg.API_PORT)
    logger.info("Gradio will be available at http://%s:%d", cfg.GRADIO_HOST, cfg.GRADIO_PORT)

    # Start FastAPI in background thread
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    logger.info("FastAPI started on port %d", cfg.API_PORT)

    # Build and launch Gradio (blocking)
    demo = build_gradio_app()
    demo.launch(
        server_name=cfg.GRADIO_HOST,
        server_port=cfg.GRADIO_PORT,
        share=cfg.GRADIO_SHARE,
        show_error=True,
        favicon_path=None,
    )

