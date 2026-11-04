"""
SignBridge AI — Phase 7, 8, 9, 10: Unified Live ISL Recognition and Character Builder Interface.
Focuses exclusively on ISL English Alphabet (A–Z, 26 classes) and Numbers (0–9, 10 classes).

Architecture & Features:
- RECOGNITION MODES:
  1. "Alphabet (A–Z)": 26 ISL fingerspelling classes evaluated on 30-frame sequence window.
  2. "Numbers (0–9)": 10 ISL number classes with syntactic verification.
  3. "Finger Count (Diagnostic)": Invariant 3D count of extended fingers (0–5).
- CAMERA DISPLAY: Exactly ONE primary camera stream showing video with green landmark dots and blue connections.
- CHARACTER TEXT BUILDER:
  - Assembles character-by-character readable text (e.g. "HELLO123").
  - Buttons: "➕ Add Character", "⌫ Backspace", "🗑 Clear Text", "🔊 Speak Text (TTS)".
  - Auto-Add Toggle: Requires sign stability for 6 frames + neutral hand release before repeating same character.
- PER-SESSION STATE ISOLATION:
  - Uses Gradio gr.State() to manage independent LivePipeline instances per browser connection.
  - Eliminates cross-user text contamination and global mutable state leaks.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gradio as gr

from backend.vision.live_pipeline import LivePipeline
from backend.translation.translator import TranslationService
from backend.audio.text_to_speech import TTSService
import config as cfg

logger = logging.getLogger(__name__)

# Fallback default pipeline for tests and headless calls
_global_fallback_pipeline: Optional[LivePipeline] = None
_translation_service: Optional[TranslationService] = None
_tts_service: Optional[TTSService] = None

_lang_map = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
}


def _get_fallback_pipeline() -> LivePipeline:
    global _global_fallback_pipeline
    if _global_fallback_pipeline is None:
        _global_fallback_pipeline = LivePipeline(
            mode="Alphabet (A–Z)",
            sequence_length=30,
            repeat_cooldown=1.0,
        )
    return _global_fallback_pipeline


def _resolve_pipeline(session_pipeline: Optional[LivePipeline]) -> LivePipeline:
    if session_pipeline is not None and isinstance(session_pipeline, LivePipeline):
        return session_pipeline
    return _get_fallback_pipeline()


def _get_translation_service() -> TranslationService:
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService(
            libre_url=cfg.LIBRE_TRANSLATE_URL,
            libre_api_key=cfg.LIBRE_TRANSLATE_API_KEY,
            offline_dict=cfg.ISL_PHRASE_TRANSLATIONS,
        )
    return _translation_service


def _get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(
            provider=cfg.TTS_PROVIDER,
            voice_map=cfg.EDGE_TTS_VOICES,
            gtts_lang_map=cfg.GTTS_LANG_MAP,
        )
    return _tts_service


# ── Streaming Frame Callback (Per-Session State) ─────────────────────────────

def on_camera_stream(
    frame: Optional[np.ndarray],
    current_sentence: str,
    mode: str,
    auto_add: bool,
    session_pipeline: Optional[LivePipeline] = None,
) -> Tuple[Optional[np.ndarray], str, str, str, str, str, str, str, str, Any]:
    """
    Continuous streaming callback from Gradio webcam component.
    Processes the frame through session-isolated LivePipeline.
    """
    pipeline = _resolve_pipeline(session_pipeline)
    res = pipeline.process_frame(
        frame=frame,
        current_sentence=current_sentence,
        mode=mode,
        auto_add=auto_add,
        is_bgr=False,
    )

    return (
        res["annotated_frame"],
        res["tracking_status"],
        res["current_sign"],
        res["confidence"],
        res["finger_count"],
        res["sentence"],
        res["gloss_sequence"],
        res["latency"],
        res["hud_status"],
        pipeline,
    )


# ── Button Actions (Character Builder & Pipeline Controls) ───────────────────

def on_start_recognition(session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, Any]:
    pipeline = _resolve_pipeline(session_pipeline)
    status = pipeline.start_recognition()
    return status, pipeline


def on_pause_recognition(session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, Any]:
    pipeline = _resolve_pipeline(session_pipeline)
    status = pipeline.pause_recognition()
    return status, pipeline


def on_stop_and_clear(session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, str, str, str, str, str, Any]:
    pipeline = _resolve_pipeline(session_pipeline)
    status, sent, gloss, curr, conf = pipeline.stop_and_clear()
    pipeline.character_builder.clear()
    return status, "", "", "—", "—", "—", pipeline


def on_add_character(session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, str, Any]:
    """Manually add the current recognized character to the text buffer."""
    pipeline = _resolve_pipeline(session_pipeline)
    sign = pipeline._last_pred_sign
    if sign and sign not in ("—", "Unknown Sign", "Collecting sign frames...", "Model Not Available"):
        # Strip descriptive prefixes like "Letter " or "Number "
        clean = sign
        for prefix in ("Letter ", "Number ", "ISL Number: ", "Count: "):
            if clean.startswith(prefix):
                clean = clean[len(prefix):].split()[0]
        pipeline.character_builder.add_character(clean)
    curr_text = pipeline.character_builder.text
    return curr_text, " → ".join(pipeline.character_builder.gloss_sequence), pipeline


def on_backspace(session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, str, Any]:
    """Remove the trailing character from the text buffer."""
    pipeline = _resolve_pipeline(session_pipeline)
    pipeline.character_builder.backspace()
    curr_text = pipeline.character_builder.text
    return curr_text, " → ".join(pipeline.character_builder.gloss_sequence), pipeline


def on_clear_text(session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, str, Any]:
    """Clear all accumulated characters."""
    pipeline = _resolve_pipeline(session_pipeline)
    pipeline.character_builder.clear()
    return "", "", pipeline


def on_mode_change(mode: str, session_pipeline: Optional[LivePipeline] = None) -> Tuple[str, Any]:
    pipeline = _resolve_pipeline(session_pipeline)
    pipeline.set_mode(mode)
    return f"Active mode: {mode}", pipeline


def on_translate_text(text: str, target_language: str) -> str:
    if not text or not text.strip():
        return "No text to translate."
    lang_code = _lang_map.get(target_language, "ta")
    svc = _get_translation_service()
    res = svc.translate(text, "en", lang_code)
    return res.get("translated_text", "")


def on_speak_text(text: str) -> Optional[str]:
    """Speak text via local Python TTS (English)."""
    if not text or not text.strip():
        return None
    tts = _get_tts_service()
    path, err = tts.synthesize(text, "en")
    if err:
        logger.warning("TTS error (en): %s", err)
    return path


def on_speak_translated(translated_text: str, target_language: str) -> Optional[str]:
    """Speak translated text via local Python TTS in target language."""
    if not translated_text or not translated_text.strip():
        return None
    lang_code = _lang_map.get(target_language, "ta")
    tts = _get_tts_service()
    path, err = tts.synthesize(translated_text, lang_code)
    if err:
        logger.warning("TTS error (%s): %s", lang_code, err)
    return path


def on_save_to_history(text: str, translated: str, target_language: str, session_pipeline: Optional[LivePipeline] = None) -> str:
    if not text or not text.strip():
        return "⚠ No text to save."
    try:
        from backend.database.database import get_db
        db = get_db()
        pipeline = _resolve_pipeline(session_pipeline)
        gloss_seq = [c for c in text]
        lang_code = _lang_map.get(target_language, "ta")
        rec_id = db.save_conversation(
            gloss_sequence=gloss_seq,
            english_text=text,
            translated_text=translated,
            language_code=lang_code,
            language_name=target_language,
        )
        return f"✅ Saved record (ID: {rec_id[:8]}...)"
    except Exception as exc:
        logger.error("Save to history error: %s", exc)
        return f"⚠ Save failed: {exc}"


# ── UI Layout ─────────────────────────────────────────────────────────────────

def build_translator_tab():
    """Build the unified Live ISL Alphabet & Numbers Translator tab."""
    with gr.Tab("🎥 Live ISL Translator"):
        gr.Markdown(
            "## 🤟 Real-Time Indian Sign Language (ISL) Recognition\n"
            "*Accurate recognition of **English Alphabet (A–Z)** and **Numbers (0–9)**. "
            "Assemble recognized characters into readable text with instant Text-to-Speech and translation.*"
        )

        # Per-session pipeline state to ensure complete user isolation
        session_pipeline = gr.State(value=lambda: LivePipeline(mode="Alphabet (A–Z)"))

        # Recognition Mode Selector
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["Alphabet (A–Z)", "Numbers (0–9)"],
                value="Alphabet (A–Z)",
                label="🎯 Recognition Mode",
                info="Alphabet (A–Z): ISL fingerspelling (26 classes) | Numbers (0–9): ISL digits (10 classes)",
            )

        with gr.Row():
            # ── LEFT COLUMN: CAMERA FEED & TRACKING TELEMETRY ──
            with gr.Column(scale=5):
                gr.Markdown("### 📷 Live Camera Feed")

                # EXACTLY ONE camera output: Annotated video with green landmark dots and blue connections
                camera_annotated = gr.Image(
                    label="Live Camera Stream with Green Landmarks & Skeleton",
                    interactive=False,
                )

                # Hardware webcam capture component collapsed in device settings accordion
                with gr.Accordion("⚙️ Webcam Hardware Source (Click here to select camera / grant permission)", open=False):
                    camera_input = gr.Image(
                        label="Hardware Webcam Capture (Start Stream Here)",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        mirror_webcam=False,
                    )

                tracking_status_box = gr.Textbox(
                    label="Hand Tracking & Landmark Telemetry",
                    value="Waiting for camera...",
                    lines=8,
                    interactive=False,
                )

                auto_add_checkbox = gr.Checkbox(
                    label="⚡ Auto-Add Stabilized Characters to Text",
                    value=False,
                    info="When unchecked (default), click 'Add Character'. When checked, adds characters held steady for 6+ frames with mandatory neutral release.",
                )

            # ── RIGHT COLUMN: PREDICTION & CHARACTER TEXT BUILDER ──
            with gr.Column(scale=5):
                gr.Markdown("### 🔤 Recognition Output")

                with gr.Row():
                    current_sign_box = gr.Textbox(
                        label="Current Prediction",
                        value="—",
                        interactive=False,
                        scale=3,
                    )
                    confidence_box = gr.Textbox(
                        label="Confidence",
                        value="—",
                        interactive=False,
                        scale=2,
                    )
                    finger_count_box = gr.Textbox(
                        label="Extended Fingers",
                        value="—",
                        interactive=False,
                        scale=2,
                    )

                with gr.Row():
                    latency_box = gr.Textbox(
                        label="Inference Latency",
                        value="0 ms",
                        interactive=False,
                        scale=3,
                    )
                    recognition_status_box = gr.Textbox(
                        label="Pipeline Status",
                        value="Ready (Alphabet Mode active)",
                        interactive=False,
                        scale=2,
                    )

                gr.Markdown("### 📝 Character Text Builder")
                sentence_box = gr.Textbox(
                    label="Accumulated Text (e.g. HELLO123)",
                    placeholder="Recognized characters appear here...",
                    interactive=True,
                    lines=2,
                )

                gloss_box = gr.Textbox(
                    label="Character History",
                    placeholder="Recent character additions...",
                    interactive=False,
                    lines=2,
                )

                # Character Action Buttons
                with gr.Row():
                    btn_add_char = gr.Button("➕ Add Character", variant="primary", scale=2)
                    btn_backspace = gr.Button("⌫ Backspace", scale=1)
                    btn_clear_text = gr.Button("🗑 Clear Text", scale=1)

        # ── BOTTOM CONTROLS: ACTION BUTTONS, TRANSLATION & SPEECH ──
        gr.Markdown("### ⚙️ Pipeline Controls & Text-to-Speech")
        with gr.Row():
            btn_start = gr.Button("▶ Start Recognition", variant="primary", scale=2)
            btn_pause = gr.Button("⏸ Pause", scale=1)
            btn_stop = gr.Button("⏹ Stop & Reset", scale=1)
            btn_speak_direct = gr.Button("🔊 Speak Text (TTS)", variant="secondary", scale=2)

        gr.Markdown("### 🌐 Multilingual Translation & Spoken Speech")
        with gr.Row():
            lang_select = gr.Dropdown(
                choices=["Tamil", "Hindi", "Telugu", "Malayalam", "Kannada", "English"],
                value="Tamil",
                label="Target Language",
                scale=2,
            )
            btn_translate = gr.Button("🌐 Translate", variant="primary", scale=1)
            btn_speak_translated = gr.Button("🔊 Speak Translation", scale=2)

        with gr.Row():
            translated_box = gr.Textbox(
                label="Translation Output",
                interactive=False,
                lines=2,
                scale=3,
            )
            audio_player = gr.Audio(
                label="Spoken Audio Output",
                autoplay=True,
                scale=2,
            )

        with gr.Row():
            btn_save = gr.Button("💾 Save to History", variant="secondary", scale=1)
            save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)

        # ── Streaming and Event Wiring ──
        camera_input.stream(
            fn=on_camera_stream,
            inputs=[camera_input, sentence_box, mode_radio, auto_add_checkbox, session_pipeline],
            outputs=[
                camera_annotated,
                tracking_status_box,
                current_sign_box,
                confidence_box,
                finger_count_box,
                sentence_box,
                gloss_box,
                latency_box,
                recognition_status_box,
                session_pipeline,
            ],
            concurrency_limit=1,
        )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio, session_pipeline],
            outputs=[recognition_status_box, session_pipeline],
        )

        btn_add_char.click(
            fn=on_add_character,
            inputs=[session_pipeline],
            outputs=[sentence_box, gloss_box, session_pipeline],
        )

        btn_backspace.click(
            fn=on_backspace,
            inputs=[session_pipeline],
            outputs=[sentence_box, gloss_box, session_pipeline],
        )

        btn_clear_text.click(
            fn=on_clear_text,
            inputs=[session_pipeline],
            outputs=[sentence_box, gloss_box, session_pipeline],
        )

        btn_start.click(
            fn=on_start_recognition,
            inputs=[session_pipeline],
            outputs=[recognition_status_box, session_pipeline],
        )

        btn_pause.click(
            fn=on_pause_recognition,
            inputs=[session_pipeline],
            outputs=[recognition_status_box, session_pipeline],
        )

        btn_stop.click(
            fn=on_stop_and_clear,
            inputs=[session_pipeline],
            outputs=[recognition_status_box, sentence_box, gloss_box, current_sign_box, confidence_box, finger_count_box, session_pipeline],
        )

        btn_speak_direct.click(
            fn=on_speak_text,
            inputs=[sentence_box],
            outputs=[audio_player],
        )

        btn_translate.click(
            fn=on_translate_text,
            inputs=[sentence_box, lang_select],
            outputs=[translated_box],
        )

        btn_speak_translated.click(
            fn=on_speak_translated,
            inputs=[translated_box, lang_select],
            outputs=[audio_player],
        )

        btn_save.click(
            fn=on_save_to_history,
            inputs=[sentence_box, translated_box, lang_select, session_pipeline],
            outputs=[save_status],
        )
