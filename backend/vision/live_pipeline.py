"""
SignBridge AI — Phase 5, 7, 8: Unified Live Camera & Multi-Mode ISL Recognition Pipeline.
Centralized, thread-safe production pipeline for real-time camera tracking and ISL recognition.

Key Enhancements:
1. Three Distinct Recognition Modes:
   - "Finger Count": Outputs 0, 1, 2, 3, 4, 5 with confidence, no word predictions.
   - "ISL Numbers": Outputs 0 through 9 with ISL syntactic verification.
   - "ISL Words": 40 ISL vocabulary signs with dynamic motion gating and unknown sign rejection.
2. Latency Optimization:
   - Resizes frame to 320x240 for MediaPipe Tasks HandLandmarker while drawing on full-resolution frame.
   - Stage-by-stage profiling via PerformanceMonitor (Ingest, Preprocessing, Detection, Extraction, Inference, Total, FPS).
3. Single Camera Stream:
   - Direct high-contrast rendering of green landmark dots (r=5) and blue skeleton connections.
4. Strict Sentence Builder:
   - Manual confirmation by default; auto-add requires 15 stable frames (>0.85 conf) and neutral-release gate.
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
    extract_two_hand_features,
    classify_handshape_name,
    TWO_HAND_FEATURE_DIM,
)
from backend.recognition.finger_counter import FingerCounter, FingerCountResult
from backend.recognition.isl_number_recognizer import ISLNumberRecognizer, ISLNumberResult
from backend.recognition.isl_fingerspelling_recognizer import ISLFingerspellingRecognizer
from backend.recognition.isl_word_recognizer import ISLWordRecognizer
from backend.recognition.sentence_builder import SentenceBuilder
from backend.recognition.character_builder import CharacterTextBuilder
from backend.vision.performance_monitor import PerformanceMonitor, LatencyProfile

logger = logging.getLogger(__name__)


class LivePipeline:
    """
    Centralized, thread-safe production pipeline for real-time hand camera tracking,
    finger counting, ISL number recognition, and 40-word ISL sentence building.
    """

    def __init__(
        self,
        mode: str = "ISL Words",
        sequence_length: int = 30,
        repeat_cooldown: float = 1.5,
        confidence_threshold: float = 0.38,
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
            min_confidence=0.45,
        )
        self.word_recognizer = ISLWordRecognizer(
            sequence_length=sequence_length,
            min_confidence=0.52,
            min_margin=0.15,
            min_motion_energy=0.012,
            cooldown_seconds=repeat_cooldown,
        )
        self.recognizer = self.word_recognizer
        self.sentence_builder = SentenceBuilder(
            repeat_cooldown_seconds=repeat_cooldown,
            required_stable_frames=15,
            min_auto_confidence=0.85,
        )
        self.character_builder = CharacterTextBuilder(
            required_stable_frames=6,
            min_auto_confidence=0.65,
            repeat_cooldown_seconds=repeat_cooldown,
        )
        self.perf_monitor = PerformanceMonitor()

        self.is_running: bool = True
        self._last_pred_sign: str = "—"
        self._last_conf_str: str = "—"
        self._last_finger_count_str: str = "—"
        self._frame_counter: int = 0

    def set_mode(self, new_mode: str):
        """Switch active recognition mode ('Finger Count', 'ISL Numbers', 'ISL Alphabet/Fingerspelling', 'ISL Words')."""
        clean = new_mode.strip()
        self.mode = clean
        self.word_recognizer.reset_buffer()
        self.fingerspelling_recognizer.reset_buffer()
        self.number_recognizer.reset_buffer()
        logger.info("LivePipeline: Mode set to '%s'", self.mode)

    def start_recognition(self) -> str:
        """Enable active ISL sequence classification."""
        self.is_running = True
        self.word_recognizer.reset_buffer()
        self.fingerspelling_recognizer.reset_buffer()
        self.number_recognizer.reset_buffer()
        logger.info("LivePipeline: Recognition active (Mode: %s)", self.mode)
        return f"▶ Recognition active (Mode: {self.mode})"

    def pause_recognition(self) -> str:
        """Pause ISL classification while keeping live camera video and landmarks visible."""
        self.is_running = False
        logger.info("LivePipeline: Recognition paused")
        return "⏸ Recognition paused (Hand tracking active)"

    def stop_and_clear(self) -> Tuple[str, str, str, str, str]:
        """Stop recognition, clear sentence builder and sequence buffer."""
        self.is_running = False
        self.word_recognizer.reset_buffer()
        self.fingerspelling_recognizer.reset_buffer()
        self.number_recognizer.reset_buffer()
        self.sentence_builder.clear()
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
            self.sentence_builder.accept_current_sign(self._last_pred_sign, 1.0)
        curr = self.character_builder.text or self.sentence_builder.current_sentence
        return curr, " → ".join(self.sentence_builder.gloss_sequence)

    def undo_last_sign(self) -> Tuple[str, str]:
        """Remove the most recently confirmed sign from the sentence."""
        removed = self.sentence_builder.undo()
        logger.info("LivePipeline: Undone sign: %s", removed)
        return self.sentence_builder.current_sentence, " → ".join(self.sentence_builder.gloss_sequence)

    def clear_sentence(self) -> Tuple[str, str]:
        """Clear sentence."""
        self.sentence_builder.clear()
        return "", ""

    def confirm_sentence(self, custom_text: Optional[str] = None) -> str:
        """Lock in confirmed sentence."""
        if custom_text and custom_text.strip():
            self.sentence_builder.set_user_text(custom_text.strip())
        return self.sentence_builder.confirmed_text

    def add_character(self, char: Optional[str]) -> str:
        """Append a recognized character to the text buffer."""
        if char:
            self.character_builder.add_character(char)
            self.sentence_builder.add_word(char)
        return self.character_builder.text or self.sentence_builder.current_sentence

    def backspace(self) -> str:
        """Remove trailing character from text buffer."""
        self.character_builder.backspace()
        self.sentence_builder.undo()
        return self.character_builder.text

    def clear_text(self) -> str:
        """Clear the character and sentence text buffers."""
        self.character_builder.clear()
        self.sentence_builder.clear()
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
        4. Mode-based classification (Finger Count, ISL Numbers, or ISL Words).
        5. Sentence builder auto-add stabilization and transition gating.
        6. Performance profile tracking.
        """
        t_start = time.perf_counter()
        self._frame_counter += 1

        if mode and mode != self.mode:
            self.set_mode(mode)

        self.sentence_builder.auto_add_enabled = auto_add

        # Handle missing or empty frame
        if frame is None or getattr(frame, "size", 0) == 0:
            return {
                "annotated_frame": None,
                "tracking_status": "Waiting for camera...",
                "current_sign": "—",
                "confidence": "—",
                "finger_count": "—",
                "sentence": self.sentence_builder.current_sentence or current_sentence or "",
                "gloss_sequence": " → ".join(self.sentence_builder.gloss_sequence),
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
            self.word_recognizer.reset_buffer()
            self.fingerspelling_recognizer.reset_buffer()
            self.number_recognizer.reset_buffer()
            self.sentence_builder.process_frame_auto_add(None, 0.0, hand_present=False)

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

            # MODE 1: ISL Alphabet / Fingerspelling Mode (A-Z)
            if "alphabet" in self.mode.lower() or "spelling" in self.mode.lower():
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
                    rec_let = alph_pred.get("letter")
                    rec_conf = alph_pred.get("confidence", 0.0)
                    status_text = alph_pred.get("status", "Recognizing...")

                    if rec_let:
                        pred_sign = f"Letter {rec_let}"
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
                t_inference_end = time.perf_counter()

            # MODE 2: ISL Number Mode (0-9 digits)
            elif "number" in self.mode.lower():
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

            # MODE 3: Finger Count Mode (0-5 diagnostic)
            elif "finger" in self.mode.lower():
                t_inference_start = time.perf_counter()
                pred_sign = str(fc_res.count)
                conf_str = f"{fc_res.confidence:.1%}"
                hud_status = f"Finger Count: {fc_res.count} (Confidence: {conf_str})"
                candidate_for_auto_add = pred_sign
                candidate_conf = fc_res.confidence
                t_inference_end = time.perf_counter()

            # MODE 4: ISL Word Mode (40 vocabulary words with motion gating)
            else:
                t_inference_start = time.perf_counter()
                if not self.word_recognizer.is_loaded:
                    pred_sign = "Model Not Available"
                    conf_str = "—"
                    hud_status = "Word model not available (models/isl_word_model.keras missing)"
                elif not self.is_running:
                    hud_status = "Recognition paused (Click Start Recognition)"
                    pred_sign = "—"
                    conf_str = "—"
                else:
                    self.word_recognizer.add_frame(feat_282, has_hands=True)
                    pred = self.word_recognizer.predict()

                    rec_word = pred.get("word")
                    rec_conf = pred.get("confidence", 0.0)
                    status_text = pred.get("status", "Recognizing...")

                    if rec_word:
                        pred_sign = rec_word.upper()
                        conf_str = f"{rec_conf:.1%}"
                        hud_status = f"Recognized: {pred_sign} ({conf_str})"
                        candidate_for_auto_add = pred_sign
                        candidate_conf = rec_conf
                    elif "Collecting" in status_text:
                        pred_sign = "Collecting sign frames..."
                        conf_str = f"{self.word_recognizer.buffer_count}/30"
                        hud_status = "Collecting dynamic sign motion..."
                    elif "Model Not Available" in status_text:
                        pred_sign = "Model Not Available"
                        conf_str = "—"
                        hud_status = "Word model not available"
                    elif "Unknown" in status_text or "Stationary" in status_text:
                        pred_sign = "Unknown Sign"
                        conf_str = f"{rec_conf:.1%}" if rec_conf > 0.1 else "—"
                        hud_status = status_text
                    else:
                        pred_sign = "—"
                        conf_str = "—"
                        hud_status = status_text
                t_inference_end = time.perf_counter()

        # ── Step 5: Sentence & Character Builder Update ──
        t_sentence_start = time.perf_counter()
        if auto_add:
            if candidate_for_auto_add:
                if "alphabet" in self.mode.lower() or "spelling" in self.mode.lower() or "number" in self.mode.lower():
                    self.character_builder.auto_add_enabled = True
                    self.character_builder.process_auto_frame(
                        character_candidate=candidate_for_auto_add,
                        confidence=candidate_conf,
                        hand_detected=True,
                    )
                if "word" in self.mode.lower() and locals().get("pred", {}).get("is_new_sign", False):
                    self.sentence_builder.add_word(candidate_for_auto_add, candidate_conf)
                else:
                    self.sentence_builder.process_frame_auto_add(
                        candidate_sign=candidate_for_auto_add,
                        confidence=candidate_conf,
                        hand_present=True,
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
        sentence_text = self.character_builder.text or self.sentence_builder.current_sentence or current_sentence or ""
        gloss_seq = " → ".join(self.sentence_builder.gloss_sequence)

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
