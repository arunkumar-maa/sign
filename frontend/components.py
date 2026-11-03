"""
Reusable Gradio UI components for SignBridge AI.
"""

import gradio as gr


def tracking_status_panel():
    """Returns a set of status text components for tracking indicators."""
    return gr.Textbox(
        label="Tracking Status",
        value="Camera not started",
        interactive=False,
        lines=4,
    )


def recognition_output_panel():
    """Recognition result display components."""
    sign = gr.Textbox(label="Current Sign", value="—", interactive=False)
    confidence = gr.Textbox(label="Confidence", value="—", interactive=False)
    return sign, confidence


def sentence_panel():
    """Sentence builder display components."""
    sentence = gr.Textbox(
        label="Recognized Sentence (editable)",
        placeholder="Signs will appear here...",
        interactive=True,
        lines=3,
    )
    gloss = gr.Textbox(
        label="ISL Gloss Sequence",
        interactive=False,
    )
    return sentence, gloss


def language_selector():
    """Language selection dropdown."""
    return gr.Dropdown(
        choices=["English", "Tamil", "Hindi", "Telugu", "Malayalam", "Kannada"],
        value="Tamil",
        label="Translate To",
    )


def translation_output():
    """Translation result display."""
    return gr.Textbox(
        label="Translation",
        interactive=False,
        lines=3,
    )


def audio_output():
    """Audio playback component."""
    return gr.Audio(label="Generated Speech", autoplay=False)

