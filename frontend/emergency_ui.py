"""
Emergency SOS tab.
"""

import logging
import gradio as gr

logger = logging.getLogger(__name__)

_DISCLAIMER = """
⚠️ IMPORTANT DISCLAIMER

This application does NOT call emergency services or contact anyone.
It generates text and audio to help you communicate with a person nearby.
Show this screen to someone around you, or ask them to call for help.

இந்த பயன்பாடு அவசர சேவைகளை அழைப்பதில்லை.
இந்த செய்தியை அருகிலுள்ள ஒருவருக்கு காட்டுங்கள்.
"""


def get_emergency_messages():
    from backend.services.emergency_service import EMERGENCY_MESSAGES
    return EMERGENCY_MESSAGES


def speak_emergency(message_id, lang):
    try:
        from backend.services.emergency_service import EmergencyService
        svc = EmergencyService()
        msg = svc.get_message(message_id)
        text = msg["tamil"] if lang == "Tamil" else msg["english"]
        lang_code = "ta" if lang == "Tamil" else "en"

        from backend.main import get_tts_service
        tts = get_tts_service()
        if tts is None:
            return None, "⚠ TTS service unavailable."
        path, error = tts.synthesize(text, lang_code)
        if error:
            return None, f"⚠ {error}"
        return path, f"✅ Speaking: {text}"
    except Exception as exc:
        return None, f"⚠ Error: {exc}"


def build_emergency_tab():
    with gr.Tab("🆘 Emergency SOS"):
        gr.Markdown(f"""
# 🆘 Emergency SOS Communication
## DISCLAIMER
{_DISCLAIMER}
        """)

        MSGS = [
            ("medical_help", "🏥 I need medical help", "எனக்கு மருத்துவ உதவி தேவை"),
            ("call_ambulance", "🚑 Call an ambulance", "ஆம்புலன்ஸ் அழையுங்கள்"),
            ("in_danger", "⚠️ I am in danger", "நான் ஆபத்தில் இருக்கிறேன்"),
            ("contact_family", "👨‍👩‍👧 Please contact my family", "என் குடும்பத்தை தொடர்பு கொள்ளுங்கள்"),
            ("cannot_hear", "🦻 I cannot hear you", "என்னால் உங்களுக்கு கேட்கவில்லை"),
        ]

        selected_id = gr.State("medical_help")
        display_box = gr.Textbox(
            label="Emergency Message",
            value="🏥 I need medical help.\nஎனக்கு மருத்துவ உதவி தேவை.",
            interactive=False,
            lines=4,
            elem_id="emergency-display",
        )
        lang_toggle = gr.Radio(choices=["English", "Tamil", "Both"], value="Both", label="Language")
        audio_out = gr.Audio(label="Emergency Audio")
        speak_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Select Emergency Message:")
        with gr.Row():
            for msg_id, en, ta in MSGS:
                def make_select(mid, e, t):
                    def select_fn():
                        return mid, f"🚨 {e}\n\n{t}"
                    return select_fn

                btn = gr.Button(f"{en}", size="lg")
                btn.click(
                    fn=make_select(msg_id, en, ta),
                    outputs=[selected_id, display_box],
                )

        with gr.Row():
            btn_speak_en = gr.Button("🔊 Speak in English", variant="primary")
            btn_speak_ta = gr.Button("🔊 Speak in Tamil", variant="primary")

        btn_speak_en.click(
            fn=lambda mid: speak_emergency(mid, "English"),
            inputs=[selected_id],
            outputs=[audio_out, speak_status],
        )
        btn_speak_ta.click(
            fn=lambda mid: speak_emergency(mid, "Tamil"),
            inputs=[selected_id],
            outputs=[audio_out, speak_status],
        )

