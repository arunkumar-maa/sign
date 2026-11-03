"""
SignBridge AI — Phase 5, 6, 7, 9: Unified Live ISL Recognition and Translation Interface.
Implements the camera-first single-screen layout:
- RECOGNITION MODES: "Finger Count" (default), "ISL Numbers", "ISL Words".
- CAMERA DISPLAY: Exactly ONE camera stream (annotated stream with green dots & blue skeleton).
  Webcam hardware capture is placed inside a collapsed device configuration accordion.
- RIGHT COLUMN: Current Sign, Confidence, Finger Count, Latency breakdown, Sentence, Glosses.
- SENTENCE CONTROLS: Accept Current Sign, Undo Last Word, Clear Sentence, Auto-add toggle (default off).
- TRANSLATION & SPEECH: Multilingual translation and TTS in Tamil, Hindi, Telugu, Malayalam, Kannada, English.
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

# Persistent singletons across streaming frames
_pipeline: Optional[LivePipeline] = None
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


def _get_pipeline() -> LivePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivePipeline(
            mode="ISL Words",
            sequence_length=30,
            repeat_cooldown=1.5,
        )
    return _pipeline


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


# ── Streaming Frame Callback ──────────────────────────────────────────────────

def on_camera_stream(
    frame: Optional[np.ndarray],
    current_sentence: str,
    mode: str,
    auto_add: bool,
) -> Tuple[Optional[np.ndarray], str, str, str, str, str, str, str, str]:
    """
    Continuous streaming callback from Gradio webcam component.
    Processes the frame through LivePipeline and returns all updated UI elements.
    """
    pipeline = _get_pipeline()
    # Gradio delivers frames in RGB format (is_bgr=False)
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
    )


# ── Button Actions ────────────────────────────────────────────────────────────

def on_start_recognition() -> str:
    pipeline = _get_pipeline()
    return pipeline.start_recognition()


def on_pause_recognition() -> str:
    pipeline = _get_pipeline()
    return pipeline.pause_recognition()


def on_stop_and_clear() -> Tuple[str, str, str, str, str, str]:
    pipeline = _get_pipeline()
    status, sent, gloss, curr, conf = pipeline.stop_and_clear()
    return status, sent, gloss, curr, conf, "—"


def on_accept_current_sign() -> Tuple[str, str]:
    pipeline = _get_pipeline()
    return pipeline.accept_current_sign()


def on_undo_last_sign() -> Tuple[str, str]:
    pipeline = _get_pipeline()
    return pipeline.undo_last_sign()


def on_clear_sentence() -> Tuple[str, str]:
    pipeline = _get_pipeline()
    return pipeline.clear_sentence()


def on_mode_change(mode: str) -> str:
    pipeline = _get_pipeline()
    pipeline.set_mode(mode)
    return f"Active mode: {mode}"


def on_translate_sentence(sentence_text: str, target_language: str) -> str:
    if not sentence_text or not sentence_text.strip():
        return "No sentence to translate."
    lang_code = _lang_map.get(target_language, "ta")
    svc = _get_translation_service()
    res = svc.translate(sentence_text, "en", lang_code)
    return res.get("translated_text", "")


def on_speak_english(sentence_text: str) -> Optional[str]:
    if not sentence_text or not sentence_text.strip():
        return None
    tts = _get_tts_service()
    path, err = tts.synthesize(sentence_text, "en")
    if err:
        logger.warning("TTS error (en): %s", err)
    return path


def on_speak_translated(translated_text: str, target_language: str) -> Optional[str]:
    if not translated_text or not translated_text.strip():
        return None
    lang_code = _lang_map.get(target_language, "ta")
    tts = _get_tts_service()
    path, err = tts.synthesize(translated_text, lang_code)
    if err:
        logger.warning("TTS error (%s): %s", lang_code, err)
    return path


def on_save_to_history(sentence: str, translated: str, target_language: str) -> str:
    if not sentence or not sentence.strip():
        return "⚠ No sentence to save."
    try:
        from backend.database.database import get_db
        db = get_db()
        pipeline = _get_pipeline()
        gloss_seq = pipeline.sentence_builder.gloss_sequence
        lang_code = _lang_map.get(target_language, "ta")
        rec_id = db.save_conversation(
            gloss_sequence=gloss_seq,
            english_text=sentence,
            translated_text=translated,
            language_code=lang_code,
            language_name=target_language,
        )
        return f"✅ Saved conversation (ID: {rec_id[:8]}...)"
    except Exception as exc:
        logger.error("Save to history error: %s", exc)
        return f"⚠ Save failed: {exc}"


# ── UI Layout ─────────────────────────────────────────────────────────────────

def build_translator_tab():
    """Build the unified Live ISL Translator tab."""
    with gr.Tab("🎥 Live ISL Translator"):
        gr.Markdown(
            "## 🤟 Real-Time Indian Sign Language (ISL) Recognition & Translation\n"
            "*Webcam streams directly inside SignBridge AI with MediaPipe green landmark dots, blue finger connections, decoupled finger counting, ISL number recognition, 40-word ISL classification, sentence builder, and multilingual speech.*"
        )

        # Recognition Mode Selector
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["ISL Words", "ISL Alphabet/Fingerspelling", "ISL Numbers", "Finger Count"],
                value="ISL Words",
                label="🎯 Recognition Mode",
                info="ISL Words: 40 vocabulary signs | ISL Alphabet/Fingerspelling: A-Z letters | ISL Numbers: 0-9 digits | Finger Count: 0-5 diagnostic",
            )

        with gr.Row():
            # ── LEFT COLUMN: SINGLE CAMERA DISPLAY & TRACKING TELEMETRY ──
            with gr.Column(scale=5):
                gr.Markdown("### 📷 Live Camera Feed")

                # EXACTLY ONE camera output visible: Annotated stream with green dots & blue skeleton
                camera_annotated = gr.Image(
                    label="Live Camera Stream with Green Landmarks & Skeleton",
                    interactive=False,
                )

                # Hardware webcam capture component collapsed in device settings accordion
                with gr.Accordion("⚙️ Webcam Hardware Source (Click here to grant camera permission / select device)", open=False):
                    camera_input = gr.Image(
                        label="Hardware Webcam Capture (Start Stream Here)",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        mirror_webcam=True,
                    )

                tracking_status_box = gr.Textbox(
                    label="Hand Tracking & Handshape Status",
                    value="Waiting for camera...",
                    lines=8,
                    interactive=False,
                )

                auto_add_checkbox = gr.Checkbox(
                    label="⚡ Auto-Add Verified Signs to Sentence",
                    value=False,  # Phase 6 Rule 1: Manual confirmation by default
                    info="When unchecked (default), press 'Accept Current Sign'. When checked, adds stable signs (>0.85 conf, 15 frames) with neutral release gate.",
                )

            # ── RIGHT COLUMN: RECOGNITION OUTPUT & SENTENCE BUILDER ──
            with gr.Column(scale=5):
                gr.Markdown("### 🔤 ISL Recognition Output")

                with gr.Row():
                    current_sign_box = gr.Textbox(
                        label="Current Sign / Word",
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
                        label="Finger Count (0-5)",
                        value="—",
                        interactive=False,
                        scale=2,
                    )

                with gr.Row():
                    latency_box = gr.Textbox(
                        label="Inference Latency Profile",
                        value="0 ms",
                        interactive=False,
                        scale=3,
                    )
                    recognition_status_box = gr.Textbox(
                        label="Recognition Status",
                        value="Ready (Finger Count Mode active)",
                        interactive=False,
                        scale=2,
                    )

                gr.Markdown("### 📝 Recognized Sentence")
                sentence_box = gr.Textbox(
                    label="Recognized Sentence (editable)",
                    placeholder="Recognized signs will appear here...",
                    interactive=True,
                    lines=3,
                )

                gloss_box = gr.Textbox(
                    label="ISL Gloss Sequence",
                    placeholder="Gloss sequence (e.g. HELLO → FRIEND)...",
                    interactive=False,
                    lines=2,
                )

                # Quick Sentence Action Buttons
                with gr.Row():
                    btn_accept = gr.Button("➕ Accept Current Sign", variant="primary", scale=2)
                    btn_undo = gr.Button("↩ Undo Last Word", scale=1)
                    btn_clear_sent = gr.Button("🗑 Clear Sentence", scale=1)

        # ── BOTTOM CONTROLS: ACTION BUTTONS, TRANSLATION & SPEECH ──
        gr.Markdown("### ⚙️ Pipeline Controls")
        with gr.Row():
            btn_start = gr.Button("▶ Start Recognition", variant="primary", scale=2)
            btn_pause = gr.Button("⏸ Pause", scale=1)
            btn_stop = gr.Button("⏹ Stop & Reset", scale=1)

        gr.Markdown("### 🌐 Multilingual Translation & Text-to-Speech")
        with gr.Row():
            lang_select = gr.Dropdown(
                choices=["Tamil", "Hindi", "Telugu", "Malayalam", "Kannada", "English"],
                value="Tamil",
                label="Target Language",
                scale=2,
            )
            btn_translate = gr.Button("🌐 Translate", variant="primary", scale=1)
            btn_speak_en = gr.Button("🔊 Speak English", scale=2)
            btn_speak_ta = gr.Button("🔊 Speak Translation", scale=2)

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
            inputs=[camera_input, sentence_box, mode_radio, auto_add_checkbox],
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
            ],
            concurrency_limit=1,
        )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[recognition_status_box],
        )

        btn_accept.click(
            fn=on_accept_current_sign,
            outputs=[sentence_box, gloss_box],
        )

        btn_undo.click(
            fn=on_undo_last_sign,
            outputs=[sentence_box, gloss_box],
        )

        btn_clear_sent.click(
            fn=on_clear_sentence,
            outputs=[sentence_box, gloss_box],
        )

        btn_start.click(
            fn=on_start_recognition,
            outputs=[recognition_status_box],
        )

        btn_pause.click(
            fn=on_pause_recognition,
            outputs=[recognition_status_box],
        )

        btn_stop.click(
            fn=on_stop_and_clear,
            outputs=[recognition_status_box, sentence_box, gloss_box, current_sign_box, confidence_box, finger_count_box],
        )

        btn_translate.click(
            fn=on_translate_sentence,
            inputs=[sentence_box, lang_select],
            outputs=[translated_box],
        )

        btn_speak_en.click(
            fn=on_speak_english,
            inputs=[sentence_box],
            outputs=[audio_player],
        )

        btn_speak_ta.click(
            fn=on_speak_translated,
            inputs=[translated_box, lang_select],
            outputs=[audio_player],
        )

        btn_save.click(
            fn=on_save_to_history,
            inputs=[sentence_box, translated_box, lang_select],
            outputs=[save_status],
        )
