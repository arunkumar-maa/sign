"""
Two-Way Communication tab for SignBridge AI.
Section A: Sign to Speech  |  Section B: Speech/Text to ISL
"""

import logging
import gradio as gr

logger = logging.getLogger(__name__)

_lang_map = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
}


def transcribe_audio(audio_file, language):
    if audio_file is None:
        return "No audio provided."
    try:
        from backend.main import get_stt_service
        svc = get_stt_service()
        if svc is None:
            return "⚠ Speech-to-text service unavailable."
        lang_code = _lang_map.get(language, None) if language != "Auto-detect" else None
        result = svc.transcribe(audio_file, language=lang_code)
        if result.get("method") == "error":
            return f"⚠ {result.get('error', 'Transcription failed.')}"
        detected = result.get("language", "unknown")
        return f"{result['text']}\n(Detected language: {detected})"
    except Exception as exc:
        logger.error("STT error: %s", exc)
        return f"⚠ Error: {exc}"


def lookup_isl_gloss(text):
    if not text or not text.strip():
        return "No text to look up."
    import config as cfg
    upper = text.strip().upper()
    gloss_map = cfg.ISL_PHRASE_TRANSLATIONS.get("en", {})
    for phrase in gloss_map:
        if phrase in upper:
            return f"✅ Found ISL gloss match: **{phrase}**"
    return f"⚠ No exact ISL gloss match found for: \"{text.strip()}\"\nISL animation is not available for this sentence."


def get_animation_for_text(text):
    if not text or not text.strip():
        return None, "No text provided."
    upper = text.strip().upper()
    try:
        from backend.main import get_animation_mapper
        mapper = get_animation_mapper()
        if mapper is None:
            return None, "⚠ Animation mapper unavailable."
        result = mapper.get_clip(upper)
        if result["available"]:
            return result["path"], f"✅ {result['message']} (source: {result['source']})"
        return None, f"⚠ {result['message']}"
    except Exception as exc:
        return None, f"⚠ Error: {exc}"


def speak_text_b2b(text, language):
    if not text or not text.strip():
        return None
    try:
        from backend.main import get_tts_service
        tts = get_tts_service()
        if tts is None:
            return None
        lang_code = _lang_map.get(language, "en")
        path, _ = tts.synthesize(text, lang_code)
        return path
    except Exception:
        return None


def build_communication_tab():
    with gr.Tab("🔄 Two-Way Communication"):
        gr.Markdown("## Two-Way Communication\nSign ↔ Speech in both directions.")

        with gr.Tabs():
            # Section A: Sign to Speech
            with gr.Tab("🤟 Sign → Speech"):
                gr.Markdown("""
**Use the Live Translator tab for real-time sign recognition.**
This section shows the full pipeline result once you have confirmed a sentence there.
Enter the confirmed English sentence below to translate and speak it.
                """)
                with gr.Row():
                    with gr.Column():
                        sign_sentence = gr.Textbox(
                            label="Confirmed English Sentence",
                            placeholder="Paste or type the confirmed sentence...",
                            lines=3,
                        )
                        lang_a = gr.Dropdown(
                            choices=["English", "Tamil", "Hindi"],
                            value="Tamil",
                            label="Target Language",
                        )
                        with gr.Row():
                            btn_translate_a = gr.Button("🌐 Translate")
                            btn_speak_a = gr.Button("🔊 Speak")

                    with gr.Column():
                        translated_a = gr.Textbox(label="Translation", interactive=False, lines=3)
                        audio_a = gr.Audio(label="Speech Output")
                        anim_info_a = gr.Textbox(label="Avatar Animation", interactive=False)

                def translate_a(text, lang):
                    if not text.strip():
                        return ""
                    lc = _lang_map.get(lang, "ta")
                    from backend.main import get_translation_service
                    svc = get_translation_service()
                    if svc is None:
                        return "⚠ Translation unavailable."
                    r = svc.translate(text, "en", lc)
                    if r.get("method") == "error":
                        return f"⚠ {r.get('error')}"
                    return r.get("translated_text", "")

                btn_translate_a.click(fn=translate_a, inputs=[sign_sentence, lang_a], outputs=[translated_a])
                btn_speak_a.click(fn=speak_text_b2b, inputs=[sign_sentence, gr.State("English")], outputs=[audio_a])

            # Section B: Speech/Text → ISL
            with gr.Tab("🎙️ Speech/Text → ISL"):
                gr.Markdown("""
Type a message or upload an audio recording. The system will:
1. Transcribe speech if needed
2. Look up the ISL gloss in the verified phrase database
3. Play a verified animation clip (if available)

**Note:** ISL animation is only shown for verified, documented signs.
Unsupported sentences will show a clear message.
                """)
                with gr.Row():
                    with gr.Column():
                        text_input_b = gr.Textbox(
                            label="Text Input",
                            placeholder="Type a message...",
                            lines=3,
                        )
                        audio_input_b = gr.Audio(
                            label="Or upload / record audio",
                            type="filepath",
                            sources=["microphone", "upload"],
                        )
                        lang_b = gr.Dropdown(
                            choices=["Auto-detect", "English", "Tamil", "Hindi"],
                            value="Auto-detect",
                            label="Audio Language",
                        )
                        with gr.Row():
                            btn_transcribe = gr.Button("📝 Transcribe Audio")
                            btn_lookup = gr.Button("🔍 Look Up ISL Gloss")
                            btn_animate = gr.Button("▶ Show Animation")

                    with gr.Column():
                        transcript_b = gr.Textbox(label="Transcript / Text", interactive=True, lines=3)
                        gloss_result = gr.Textbox(label="ISL Gloss Match Result", interactive=False)
                        video_b = gr.Video(label="ISL Animation (verified signs only)")
                        anim_msg_b = gr.Textbox(label="Animation Status", interactive=False)

                btn_transcribe.click(fn=transcribe_audio, inputs=[audio_input_b, lang_b], outputs=[transcript_b])
                btn_lookup.click(fn=lookup_isl_gloss, inputs=[transcript_b], outputs=[gloss_result])

                def get_anim(text):
                    path, msg = get_animation_for_text(text)
                    return path, msg

                btn_animate.click(fn=get_anim, inputs=[transcript_b], outputs=[video_b, anim_msg_b])

