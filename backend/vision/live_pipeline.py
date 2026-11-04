"""
SignBridge AI — Phase 7, 8, 9, 10: Unified Live Camera & Multi-Mode ISL Recognition Pipeline.
Centralized, thread-safe production pipeline for real-time camera tracking and ISL recognition.
Focuses exclusively on English Alphabet Fingerspelling (A–Z) and Numbers (0–9).

Key Features:
1. Two Core Recognition Modes:
   - "Alphabet (A–Z)": 26 ISL fingerspelling classes evaluated on 30-frame sequence window.
   - "Numbers (0–9)": 10 ISL number classes with syntactic verification.
2. Latency Optimization:
   - Resizes frame to 320x240 for MediaPipe Tasks HandLandmarker while drawing on full-resolution frame.
   - Stage-by-stage profiling via PerformanceMonitor (Ingest, Preprocessing, Detection, Extraction, Inference, Total, FPS).
3. Single Camera Stream:
   - Direct high-contrast rendering of green landmark dots (r=5) and blue skeleton connections.
4. Character Text Builder:
   - Character-by-character assembly into readable strings (e.g. "HELLO123").
   - Auto-add requires stability for 6 frames + mandatory neutral release gate.
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

from backend.vision.hand_landmarks import HandLandmarker, HandLandmarkResult
from backend.vision.feature_schema import (
    extract_frame_features,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    validate_features,
)
from backend.vision.hand_pose_features import (
    classify_handshape_name,
)
from backend.recognition.finger_counter import FingerCounter
from backend.recognition.isl_number_recognizer import ISLNumberRecognizer
from backend.recognition.isl_fingerspelling_recognizer import ISLFingerspellingRecognizer
from backend.recognition.character_builder import CharacterTextBuilder
from backend.vision.performance_monitor import PerformanceMonitor, LatencyProfile

logger = logging.getLogger(__name__)


class _DummyWordRecognizer:
    """Historical placeholder shim to ensure backward compatibility without importing ISLWordRecognizer."""
    def __init__(self):
        self._buffer: List[np.ndarray] = []
        self.is_loaded: bool = False
        self.confidence_threshold: float = 0.5

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def reset_buffer(self) -> None:
        self._buffer.clear()

    def add_frame(self, *args, **kwargs) -> None:
        pass

    def predict(self) -> Dict[str, Any]:
        return {
            "status": "Model Not Available",
            "word": None,
            "confidence": 0.0,
            "is_model_available": False,
        }


class LivePipeline:
    """
    Centralized, thread-safe production pipeline for real-time hand camera tracking,
    ISL Alphabet (A–Z) and ISL Number (0–9) recognition, and Character Text Building.
    """

    def __init__(
        self,
        mode: str = "Alphabet (A–Z)",
        sequence_length: int = 30,
        repeat_cooldown: float = 1.0,
        confidence_threshold: float = 0.45,
    ):
        self.mode: str = mode
        self.confidence_threshold = confidence_threshold
        self.hand_detector = HandLandmarker(
            min_detection_confidence=0.3,
            min_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.finger_counter = FingerCounter()
        self.number_recognizer = ISLNumberRecognizer()
        self.fingerspelling_recognizer = ISLFingerspellingRecognizer(
            sequence_length=sequence_length,
            min_confidence=confidence_threshold,
        )
        self.character_builder = CharacterTextBuilder(
            required_stable_frames=6,
            min_auto_confidence=0.65,
            repeat_cooldown_seconds=repeat_cooldown,
        )

        # Backward compatibility aliases for existing caller inspections
        self.recognizer = self.fingerspelling_recognizer
        self.sentence_builder = self.character_builder
        self.word_recognizer = _DummyWordRecognizer()

        self.perf_monitor = PerformanceMonitor()

        self.is_running: bool = True
        self._last_pred_sign: str = "—"
        self._last_conf_str: str = "—"
        self._last_finger_count_str: str = "—"
        self._frame_counter: int = 0

    def set_mode(self, new_mode: str):
        """Switch active recognition mode ('Alphabet (A–Z)', 'Numbers (0–9)')."""
        clean = new_mode.strip()
        self.mode = clean
        self.fingerspelling_recognizer.reset_buffer()
        self.number_recognizer.reset_buffer()
        self.word_recognizer.reset_buffer()
        if "number" in clean.lower():
            self.recognizer = self.number_recognizer
        else:
            self.recognizer = self.fingerspelling_recognizer
        logger.info("LivePipeline: Mode set to '%s'", self.mode)

    def start_recognition(self) -> str:
        """Enable active ISL sequence classification."""
        self.is_running = True
        self.fingerspelling_recognizer.reset_buffer()
        self.number_recognizer.reset_buffer()
        self.word_recognizer.reset_buffer()
        logger.info("LivePipeline: Recognition active (Mode: %s)", self.mode)
        return f"▶ Recognition active (Mode: {self.mode})"

    def pause_recognition(self) -> str:
        """Pause ISL classification while keeping live camera video and landmarks visible."""
        self.is_running = False
        logger.info("LivePipeline: Recognition paused")
        return "⏸ Recognition paused (Hand tracking active)"

    def stop_and_clear(self) -> Tuple[str, str, str, str, str]:
        """Stop recognition, clear character text builder and sequence buffer."""
        self.is_running = False
        self.fingerspelling_recognizer.reset_buffer()
        self.number_recognizer.reset_buffer()
        self.word_recognizer.reset_buffer()
        self.character_builder.clear()
        self._last_pred_sign = "—"
        self._last_conf_str = "—"
        self._last_finger_count_str = "—"
        logger.info("LivePipeline: Stopped and cleared")
        return "⏹ Stopped & Cleared", "", "", "—", "—"

    def accept_current_sign(self) -> Tuple[str, str]:
        """User clicked 'Accept Current Sign'."""
        if self._last_pred_sign and self._last_pred_sign not in ("—", "Unknown Sign", "Collecting sign frames...", "Model Not Available"):
            clean = self._last_pred_sign
            for prefix in ("Letter ", "Number ", "ISL Number: ", "Count: "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):].split()[0]
            self.character_builder.add_character(clean)
        curr = self.character_builder.text
        return curr, " → ".join(self.character_builder.gloss_sequence)

    def undo_last_sign(self) -> Tuple[str, str]:
        """Remove the most recently confirmed character from the text."""
        self.character_builder.backspace()
        return self.character_builder.text, " → ".join(self.character_builder.gloss_sequence)

    def clear_sentence(self) -> Tuple[str, str]:
        """Clear text."""
        self.character_builder.clear()
        return "", ""

    def confirm_sentence(self, custom_text: Optional[str] = None) -> str:
        """Lock in confirmed text."""
        if custom_text and custom_text.strip():
            self.character_builder.set_text(custom_text.strip())
        return self.character_builder.confirmed_text

    def add_character(self, char: Optional[str]) -> str:
        """Append a recognized character to the text buffer."""
        if char:
            self.character_builder.add_character(char)
        return self.character_builder.text

    def backspace(self) -> str:
        """Remove trailing character from text buffer."""
        self.character_builder.backspace()
        return self.character_builder.text

    def clear_text(self) -> str:
        """Clear the character text buffer."""
        self.character_builder.clear()
        return ""

    def _extract_282_features(self, hand_result: HandLandmarkResult) -> np.ndarray:
        """
        Extract standardized 282-dim hand landmark feature vector via shared Feature Schema (v2.0.0).
        """
        feat, _ = extract_frame_features(hand_result)
        return feat

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        current_sentence: str = "",
        mode: Optional[str] = None,
        auto_add: bool = False,
        is_bgr: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single camera frame through the optimized pipeline:
        1. Downsample for MediaPipe detection (320x240) while drawing on full frame.
        2. MediaPipe Hand Landmark detection.
        3. Draw green dots and blue connections directly on frame.
        4. Mode-based classification (ISL Numbers or ISL Alphabet).
        5. Character builder auto-add stabilization and neutral-release gating.
        6. Performance profile tracking.
        """
        t_start = time.perf_counter()
        self._frame_counter += 1

        if mode and mode != self.mode:
            self.set_mode(mode)

        self.character_builder.auto_add_enabled = auto_add

        # Handle missing or empty frame
        if frame is None or getattr(frame, "size", 0) == 0:
            return {
                "annotated_frame": None,
                "tracking_status": "Waiting for camera...",
                "current_sign": "—",
                "confidence": "—",
                "finger_count": "—",
                "sentence": self.character_builder.text or current_sentence or "",
                "gloss_sequence": " → ".join(self.character_builder.gloss_sequence),
                "latency": "0 ms",
                "hud_status": "Waiting for camera",
            }

        t_ingest = time.perf_counter()

        # ── Step 1: Preprocessing & Downscaling ──
        # Resize to 320x240 for MediaPipe processing to keep latency < 35 ms
        orig_h, orig_w = frame.shape[:2]
        if orig_w > 320 or orig_h > 240:
            detection_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        else:
            detection_frame = frame

        t_prep = time.perf_counter()

        # ── Step 2: MediaPipe Tasks Hand Detection ──
        hand_result: HandLandmarkResult = self.hand_detector.detect(detection_frame, is_bgr=is_bgr)
        t_detect = time.perf_counter()

        # ── Step 3: Draw Green Dots & Blue Skeleton on Original Frame ──
        annotated_frame = self.hand_detector.draw_landmarks(frame, hand_result, is_bgr=is_bgr)

        # ── Step 4: Mode-based Feature Extraction & Inference ──
        t_extract_start = time.perf_counter()
        t_extract_end = t_extract_start
        t_inference_start = t_extract_start
        t_inference_end = t_extract_start

        num_hands = hand_result.num_hands
        pred_sign = "—"
        conf_str = "—"
        finger_count_str = "—"
        hud_status = "Waiting for hand..."
        candidate_for_auto_add: Optional[str] = None
        candidate_conf: float = 0.0

        if not hand_result.detected or num_hands == 0:
            tracking_status = (
                f"Mode: {self.mode}\n"
                f"Status: No hand detected\n"
                f"• Show your hand clearly in front of the camera\n"
                f"• Left hand tracked: NO\n"
                f"• Right hand tracked: NO\n"
                f"• Finger Count: —\n"
                f"• Handshape: None\n"
                f"• Camera FPS: {hand_result.fps:.1f}"
            )
            hud_status = "Recognition paused (No hand detected)" if not self.is_running else "No hand detected"
            self.fingerspelling_recognizer.reset_buffer()
            self.number_recognizer.reset_buffer()
            self.character_builder.process_auto_frame(None, 0.0, hand_detected=False)

        else:
            # Determine primary hand landmarks
            primary_lms = hand_result.right_landmarks if hand_result.right_hand_detected else hand_result.left_landmarks
            handedness = "Right" if hand_result.right_hand_detected else "Left"

            # Always compute robust 3D rotation-invariant finger count
            fc_res = self.finger_counter.count_fingers(primary_lms, handedness=handedness)
            finger_count_str = str(fc_res.count)
            self._last_finger_count_str = finger_count_str

            handshapes: List[str] = []
            if hand_result.right_hand_detected and hand_result.right_landmarks is not None:
                shape_r = classify_handshape_name(hand_result.right_landmarks, is_left=False)
                handshapes.append(f"Right: {shape_r}")
            if hand_result.left_hand_detected and hand_result.left_landmarks is not None:
                shape_l = classify_handshape_name(hand_result.left_landmarks, is_left=True)
                handshapes.append(f"Left: {shape_l}")
            handshape_str = ", ".join(handshapes) if handshapes else "Detected"

            total_lms = 21 * num_hands
            tracking_status = (
                f"Mode: {self.mode}\n"
                f"Status: Hand detected ({num_hands} hand{'s' if num_hands > 1 else ''})\n"
                f"• Left hand tracked: {'YES' if hand_result.left_hand_detected else 'NO'}\n"
                f"• Right hand tracked: {'YES' if hand_result.right_hand_detected else 'NO'}\n"
                f"• Fingers Extended: {fc_res.count} ({fc_res.details.get('fingers', '')})\n"
                f"• Handshape: {handshape_str}\n"
                f"• Landmarks: {total_lms}/{total_lms} (21/21 per hand)\n"
                f"• Camera FPS: {hand_result.fps:.1f}"
            )

            t_extract_start = time.perf_counter()
            feat_282 = self._extract_282_features(hand_result)
            t_extract_end = time.perf_counter()

            # MODE 1: ISL Number Mode (0-9 digits)
            if "number" in self.mode.lower():
                t_inference_start = time.perf_counter()
                if not self.number_recognizer.is_loaded:
                    pred_sign = "Model Not Available"
                    conf_str = "—"
                    hud_status = "Number model not available (models/isl_number_model.keras missing)"
                elif not self.is_running:
                    hud_status = "Recognition paused (Click Start Recognition)"
                    pred_sign = "—"
                    conf_str = "—"
                else:
                    self.number_recognizer.add_frame(feat_282, has_hands=True)
                    num_pred = self.number_recognizer.predict()
                    rec_num = num_pred.get("number")
                    rec_conf = num_pred.get("confidence", 0.0)
                    status_text = num_pred.get("status", "Recognizing...")

                    if rec_num is not None:
                        pred_sign = f"Number {rec_num}"
                        conf_str = f"{rec_conf:.1%}"
                        hud_status = f"ISL Number: {rec_num} ({conf_str})"
                        candidate_for_auto_add = str(rec_num)
                        candidate_conf = rec_conf
                    elif "Collecting" in status_text:
                        pred_sign = "Collecting sign frames..."
                        conf_str = f"{self.number_recognizer.buffer_count}/30"
                        hud_status = "Collecting number sign frames..."
                    elif "Model Not Available" in status_text:
                        pred_sign = "Model Not Available"
                        conf_str = "—"
                        hud_status = "Number model not available"
                    else:
                        pred_sign = "Unknown Sign"
                        conf_str = f"{rec_conf:.1%}" if rec_conf > 0.1 else "—"
                        hud_status = status_text
                t_inference_end = time.perf_counter()

            # MODE 2: ISL Alphabet / Fingerspelling Mode (A-Z) (Default)
            else:
                t_inference_start = time.perf_counter()
                if not self.fingerspelling_recognizer.is_loaded:
                    pred_sign = "Model Not Available"
                    conf_str = "—"
                    hud_status = "Fingerspelling model not available (models/isl_fingerspelling_model.keras missing)"
                elif not self.is_running:
                    hud_status = "Recognition paused (Click Start Recognition)"
                    pred_sign = "—"
                    conf_str = "—"
                else:
                    self.fingerspelling_recognizer.add_frame(feat_282, has_hands=True)
                    alph_pred = self.fingerspelling_recognizer.predict()
                    rec_let = alph_pred.get("letter") or alph_pred.get("word")
                    rec_conf = alph_pred.get("confidence", 0.0)
                    status_text = alph_pred.get("status", "Recognizing...")
                    is_new_sign = alph_pred.get("is_new_sign", True)

                    if rec_let:
                        pred_sign = f"Letter {rec_let}" if len(str(rec_let)) == 1 else str(rec_let).capitalize()
                        conf_str = f"{rec_conf:.1%}"
                        hud_status = f"Recognized: {pred_sign} ({conf_str})"
                        candidate_for_auto_add = rec_let
                        candidate_conf = rec_conf
                    elif "Collecting" in status_text:
                        pred_sign = "Collecting sign frames..."
                        conf_str = f"{self.fingerspelling_recognizer.buffer_count}/30"
                        hud_status = "Collecting alphabet frames..."
                    elif "Model Not Available" in status_text:
                        pred_sign = "Model Not Available"
                        conf_str = "—"
                        hud_status = "Fingerspelling model not available"
                    else:
                        pred_sign = "Unknown Sign"
                        conf_str = f"{rec_conf:.1%}" if rec_conf > 0.1 else "—"
                        hud_status = status_text

                    if "top_5" in alph_pred:
                        t5_list = [f"{item['letter']}:{item['probability']:.1%}" for item in alph_pred["top_5"][:3]]
                        tracking_status += f"\n• Top Predictions: {', '.join(t5_list)}"
                        if "margin" in alph_pred:
                            tracking_status += f" (Margin: {alph_pred['margin']:.1%})"
                t_inference_end = time.perf_counter()

        # ── Step 5: Character Builder Auto-Add Stabilization ──
        t_sentence_start = time.perf_counter()
        if auto_add and candidate_for_auto_add:
            is_new = locals().get("is_new_sign", True)
            if is_new:
                if len(str(candidate_for_auto_add)) > 1:
                    self.character_builder.add_word(candidate_for_auto_add, candidate_conf)
                else:
                    self.character_builder.auto_add_enabled = True
                    self.character_builder.process_auto_frame(
                        character_candidate=candidate_for_auto_add,
                        confidence=candidate_conf,
                        hand_detected=True,
                    )
        t_sentence_end = time.perf_counter()

        self._last_pred_sign = pred_sign
        self._last_conf_str = conf_str

        # ── Step 6: Performance Profiling ──
        t_end = time.perf_counter()
        ingest_ms = (t_ingest - t_start) * 1000.0
        prep_ms = (t_prep - t_ingest) * 1000.0
        detect_ms = (t_detect - t_prep) * 1000.0
        extract_ms = (t_extract_end - t_extract_start) * 1000.0
        inference_ms = (t_inference_end - t_inference_start) * 1000.0
        sentence_ms = (t_sentence_end - t_sentence_start) * 1000.0
        total_ms = (t_end - t_start) * 1000.0

        profile = self.perf_monitor.record_stage(
            ingest_ms=ingest_ms,
            prep_ms=prep_ms,
            detect_ms=detect_ms,
            extract_ms=extract_ms,
            inference_ms=inference_ms,
            sentence_ms=sentence_ms,
            total_ms=total_ms,
        )

        latency_summary = profile.summary_string()
        sentence_text = self.character_builder.text or current_sentence or ""
        gloss_seq = " → ".join(self.character_builder.gloss_sequence)

        return {
            "annotated_frame": annotated_frame,
            "tracking_status": tracking_status,
            "current_sign": pred_sign,
            "confidence": conf_str,
            "finger_count": finger_count_str,
            "sentence": sentence_text,
            "gloss_sequence": gloss_seq,
            "latency": latency_summary,
            "hud_status": hud_status,
            "profile": profile,
        }
