"""
Camera & Finger Detection Diagnostics Tab for SignBridge AI.
Provides real-time camera telemetry, lighting feedback, latency profiling,
and an interactive test mode for 9 diagnostic hand postures.
"""

import time
import logging
from typing import Tuple, Dict, Any

import cv2
import gradio as gr
import numpy as np

from backend.vision.hand_landmarks import HandLandmarker, HandLandmarkResult
from backend.camera.diagnostics import FingerDiagnostics, DiagnosticMetrics

logger = logging.getLogger(__name__)

# State for diagnostics
_hand_detector = None
_diag_engine = None
_last_frame_time = time.time()
_fps_rolling = 0.0


def _get_detector() -> HandLandmarker:
    global _hand_detector
    if _hand_detector is None:
        _hand_detector = HandLandmarker()
    return _hand_detector


def _get_diag_engine() -> FingerDiagnostics:
    global _diag_engine
    if _diag_engine is None:
        _diag_engine = FingerDiagnostics()
    return _diag_engine


def reset_diagnostics_tests() -> str:
    """Reset all 9 posture test status flags."""
    engine = _get_diag_engine()
    engine.reset_tests()
    return "All diagnostic posture tests reset. Present each posture to test."


def process_diagnostic_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, str, str]:
    """
    Callback for the Gradio diagnostic webcam stream.
    Gradio delivers frames in RGB format.
    """
    global _last_frame_time, _fps_rolling

    if frame is None or frame.size == 0:
        return (
            None,
            "Camera disconnected or no frame received.",
            "Lighting: Unknown",
            "### Posture Tests Status:\nWaiting for camera feed...",
        )

    # Compute rolling FPS
    curr_time = time.time()
    dt = curr_time - _last_frame_time
    _last_frame_time = curr_time
    if dt > 0.001:
        instant_fps = 1.0 / dt
        _fps_rolling = 0.85 * _fps_rolling + 0.15 * instant_fps if _fps_rolling > 0 else instant_fps

    detector = _get_detector()
    engine = _get_diag_engine()

    # Note: Gradio image type="numpy" delivers RGB
    # We pass is_bgr=False so HandLandmarker does not invert colors!
    hand_result = detector.detect(frame, is_bgr=False)

    # Annotate frame with skeleton, landmarks and handedness (in RGB color space)
    annotated = detector.draw_landmarks(frame, hand_result, is_bgr=False)

    # Evaluate camera & posture diagnostics
    metrics: DiagnosticMetrics = engine.evaluate_frame(annotated, hand_result, _fps_rolling)

    # Format telemetry text
    telemetry_str = metrics.summary_text()

    # Lighting status banner
    if "Dark" in metrics.lighting_condition:
        lighting_banner = "⚠️ **POOR LIGHTING WARNING:** Scene is too dark. Increase ambient light for accurate tracking."
    elif "Bright" in metrics.lighting_condition:
        lighting_banner = "⚠️ **OVEREXPOSURE WARNING:** Scene is washed out. Reduce intense direct glare."
    else:
        lighting_banner = "✅ **LIGHTING GOOD:** Brightness is optimal for finger landmark detection."

    # Format 9-Item checklist
    checklist_lines = ["### 🎯 Diagnostic Posture Checklist:"]
    for posture_name in engine.POSTURE_NAMES:
        passed = metrics.posture_test_results.get(posture_name, False)
        status_icon = "✅ PASSED" if passed else "⏳ WAITING"
        checklist_lines.append(f"- **{posture_name}**: `{status_icon}`")

    checklist_str = "\n".join(checklist_lines)

    return annotated, telemetry_str, lighting_banner, checklist_str


def build_diagnostics_tab():
    """Build the Camera Diagnostics tab in Gradio."""
    with gr.Tab("🔍 Camera Diagnostics"):
        gr.Markdown(
            "## 📷 Camera & Finger Detection Diagnostics\n"
            "*Real-time hardware verification, lighting analysis, MediaPipe Tasks latency, and 9-point posture testing.*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Live Camera Stream")
                diag_camera_input = gr.Image(
                    label="Camera Diagnostic Stream (Click to start)",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    mirror_webcam=True,
                )
                diag_camera_annotated = gr.Image(
                    label="Live Diagnostic Landmark Skeleton",
                    interactive=False,
                )
                lighting_display = gr.Markdown("Lighting: Initializing...")
                btn_reset = gr.Button("🔄 Reset Posture Checklist", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Hardware & Landmark Telemetry")
                telemetry_box = gr.Textbox(
                    label="Real-time Diagnostic Metrics",
                    value="Start camera to initialize telemetry...",
                    lines=9,
                    interactive=False,
                )

                gr.Markdown("### 🧪 Diagnostic Posture Verification Mode")
                gr.Markdown(
                    "Hold each posture clearly in front of the lens for 1-2 seconds to verify finger segmentation:"
                )
                checklist_display = gr.Markdown("### Posture Tests Status:\nWaiting for camera feed...")

        # Events
        diag_camera_input.stream(
            fn=process_diagnostic_frame,
            inputs=[diag_camera_input],
            outputs=[diag_camera_annotated, telemetry_box, lighting_display, checklist_display],
            concurrency_limit=1,
        )

        btn_reset.click(fn=reset_diagnostics_tests, outputs=[telemetry_box])

