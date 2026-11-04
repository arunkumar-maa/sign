import threading
import time
import uuid

import gradio as gr
import numpy as np

from .config import CATEGORIES
from .hand_landmarks import HandPoseDetector, draw_hand_overlay
from .recording import recording_manager
from .storage import sample_counts

_SHARED_DETECTOR = None
_DETECTOR_LOCK = threading.RLock()
_LAST_TIMESTAMP = 0


def _get_detector():
    global _SHARED_DETECTOR
    with _DETECTOR_LOCK:
        if _SHARED_DETECTOR is None:
            _SHARED_DETECTOR = HandPoseDetector()
        return _SHARED_DETECTOR


def _process_frame(frame, session_id):
    global _LAST_TIMESTAMP
    if frame is None:
        return None, "Waiting for webcam. Click the camera icon to start."
    try:
        rgb = np.asarray(frame, dtype=np.uint8)[:, :, :3]
        detector = _get_detector()
        with _DETECTOR_LOCK:
            timestamp = max(time.monotonic_ns() // 1_000_000, _LAST_TIMESTAMP + 1)
            _LAST_TIMESTAMP = timestamp
            result = detector.detect_video(rgb, timestamp)
        hands = getattr(result, "hand_landmarks", []) or []
        has_hands = len(hands) > 0
        record_status = recording_manager.append_frame(session_id, rgb, has_hands=has_hands)
        annotated = draw_hand_overlay(rgb, result)
        landmark_counts = [len(hand) for hand in hands]
        valid_hands = sum(1 for count in landmark_counts if count == 21)
        status = (
            f"Hands detected: {len(hands)} | Complete 21-point hands: {valid_hands} | "
            f"Landmark counts: {landmark_counts or 'none'}\n{record_status}"
        )
        return annotated, status
    except Exception as exc:
        return frame, f"Camera processing error: {type(exc).__name__}: {exc}"


def _start_recording(session_id, category, label, signer_id, consent):
    try:
        return recording_manager.start(session_id, category, label, signer_id, consent)
    except Exception as exc:
        return f"Cannot start recording: {exc}"


def _stop_recording(session_id):
    try:
        return recording_manager.stop(session_id)
    except Exception as exc:
        return f"Could not save recording: {exc}"


def _refresh_counts():
    counts = sample_counts()
    lines = ["### Saved Dataset Statistics"]

    # Alphabet statistics
    alph_counts = counts.get("alphabet", {})
    all_letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    missing_letters = [l for l in all_letters if l not in alph_counts or alph_counts[l] == 0]
    total_alph = sum(alph_counts.values())
    lines.append(f"**Alphabet (A–Z)** — Total Samples: {total_alph} across {len(alph_counts)}/26 classes.")
    if alph_counts:
        lines.append("• Recorded: " + ", ".join(f"{k}: {v}" for k, v in sorted(alph_counts.items())))
    if missing_letters:
        lines.append(f"• Missing ({len(missing_letters)}): {', '.join(missing_letters)}")

    # Number statistics
    num_counts = counts.get("number", {})
    all_digits = [str(i) for i in range(10)]
    missing_digits = [d for d in all_digits if d not in num_counts or num_counts[d] == 0]
    total_num = sum(num_counts.values())
    lines.append(f"\n**Numbers (0–9)** — Total Samples: {total_num} across {len(num_counts)}/10 classes.")
    if num_counts:
        lines.append("• Recorded: " + ", ".join(f"{k}: {v}" for k, v in sorted(num_counts.items())))
    if missing_digits:
        lines.append(f"• Missing ({len(missing_digits)}): {', '.join(missing_digits)}")

    return "\n".join(lines)


def _extract_landmarks():
    try:
        from .extraction import extract_dataset
        return extract_dataset(force=False)
    except Exception as exc:
        return f"Landmark extraction failed: {type(exc).__name__}: {exc}"


def _train(category, epochs):
    try:
        from .training import train_category
        return train_category(category, epochs=int(epochs))
    except Exception as exc:
        return f"Training failed: {type(exc).__name__}: {exc}"


def build_data_collection_tab():
    """Call this inside the existing SignBridge AI gr.Blocks context."""
    with gr.Tab("ISL Data Collection"):
        gr.Markdown(
            "## Record and label your own ISL training examples\n"
            "Collect **words, alphabet/fingerspelling and numbers separately**. "
            "The original unannotated webcam video is saved for training; the preview may show landmarks."
        )
        session_id = gr.State(value=lambda: uuid.uuid4().hex)
        with gr.Row():
            with gr.Column(scale=1):
                category = gr.Dropdown(
                    choices=[("ISL Alphabet (A–Z)", "alphabet"), ("ISL Numbers (0–9)", "number")],
                    value="alphabet", label="Recognition category"
                )
                label = gr.Textbox(label="Exact verified label", placeholder="Example: HELLO, A, TWO")
                signer_id = gr.Textbox(label="Pseudonymous signer ID", placeholder="Example: S01", value="S01")
                consent = gr.Checkbox(label="I have permission/consent to record this signer", value=False)
                with gr.Row():
                    start_button = gr.Button("Start Recording", variant="primary")
                    stop_button = gr.Button("Stop Recording", variant="stop")
                status = gr.Textbox(label="Recording and detection status", lines=4, interactive=False)
                refresh_button = gr.Button("Refresh Dataset Counts")
                counts = gr.Markdown(_refresh_counts())
            with gr.Column(scale=2):
                camera = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam input")
                annotated = gr.Image(label="Hand landmark preview", interactive=False)
                gr.Markdown(
                    "**Recording guidance:** keep the full hand visible, perform one sign per clip, "
                    "record from a consistent front-facing camera, and collect several examples from multiple signers. "
                    "Do not record bystanders without permission."
                )
        stream_kwargs = {"concurrency_limit": 1}
        import inspect
        sig = inspect.signature(camera.stream)
        if "stream_every" in sig.parameters:
            stream_kwargs["stream_every"] = 0.1
        if "time_limit" in sig.parameters:
            stream_kwargs["time_limit"] = 60

        camera.stream(
            fn=_process_frame,
            inputs=[camera, session_id],
            outputs=[annotated, status],
            **stream_kwargs,
        )
        start_button.click(
            fn=_start_recording,
            inputs=[session_id, category, label, signer_id, consent],
            outputs=[status],
        )
        stop_button.click(fn=_stop_recording, inputs=[session_id], outputs=[status])
        refresh_button.click(fn=_refresh_counts, outputs=[counts])

        gr.Markdown("## Landmark extraction and model training")
        gr.Markdown(
            "After collecting clips, extract landmarks. Then train one separate model for each category. "
            "Training requires multiple signer IDs in every category to create signer-independent test data."
        )
        with gr.Row():
            extract_button = gr.Button("Extract Hand Landmarks", variant="secondary")
            train_category = gr.Dropdown(choices=["alphabet", "number"], value="alphabet", label="Category to train")
            epochs = gr.Slider(5, 100, value=30, step=5, label="Training epochs")
            train_button = gr.Button("Train Selected Model", variant="primary")
        job_status = gr.Textbox(label="Extraction / training result", lines=7, interactive=False)
        extract_button.click(fn=_extract_landmarks, outputs=[job_status])
        train_button.click(fn=_train, inputs=[train_category, epochs], outputs=[job_status])
