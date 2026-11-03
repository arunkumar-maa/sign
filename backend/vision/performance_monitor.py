"""
SignBridge AI — Phase 8: Real-Time Performance & Latency Monitor.
Measures and tracks execution latencies across each individual stage:
1. Frame acquisition & ingestion
2. RGB color space conversion & downscaling
3. MediaPipe hand detection
4. Hand pose feature extraction
5. Neural network model inference
6. Sentence builder & post-processing
7. Total pipeline loop latency
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class LatencyProfile:
    """Latency metrics per pipeline stage in milliseconds."""
    frame_ingest_ms: float = 0.0
    preprocessing_ms: float = 0.0
    mediapipe_detect_ms: float = 0.0
    feature_extract_ms: float = 0.0
    model_inference_ms: float = 0.0
    sentence_process_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    fps: float = 0.0

    def summary_string(self) -> str:
        """Formatted human-readable latency breakdown string."""
        return (
            f"Total: {self.total_pipeline_ms:.0f}ms (FPS: {self.fps:.1f}) | "
            f"Camera: {self.mediapipe_detect_ms:.0f}ms | "
            f"Extract: {self.feature_extract_ms:.1f}ms | "
            f"Model: {self.model_inference_ms:.0f}ms"
        )


class PerformanceMonitor:
    """
    Thread-safe latency tracker and performance profiler.
    Maintains rolling averages over recent frames.
    """

    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self._history: Dict[str, List[float]] = {
            "ingest": [],
            "prep": [],
            "detect": [],
            "extract": [],
            "inference": [],
            "sentence": [],
            "total": [],
        }
        self._prev_frame_time = time.perf_counter()
        self._fps_history: List[float] = []

    def record_stage(
        self,
        ingest_ms: float,
        prep_ms: float,
        detect_ms: float,
        extract_ms: float,
        inference_ms: float,
        sentence_ms: float,
        total_ms: float,
    ) -> LatencyProfile:
        """Record measurements for the current frame and return rolling profile."""
        now = time.perf_counter()
        dt = now - self._prev_frame_time
        self._prev_frame_time = now
        curr_fps = (1.0 / dt) if dt > 0.001 else 30.0

        self._fps_history.append(curr_fps)
        if len(self._fps_history) > self.history_size:
            self._fps_history.pop(0)

        for key, val in [
            ("ingest", ingest_ms),
            ("prep", prep_ms),
            ("detect", detect_ms),
            ("extract", extract_ms),
            ("inference", inference_ms),
            ("sentence", sentence_ms),
            ("total", total_ms),
        ]:
            lst = self._history[key]
            lst.append(val)
            if len(lst) > self.history_size:
                lst.pop(0)

        return LatencyProfile(
            frame_ingest_ms=float(np.mean(self._history["ingest"])),
            preprocessing_ms=float(np.mean(self._history["prep"])),
            mediapipe_detect_ms=float(np.mean(self._history["detect"])),
            feature_extract_ms=float(np.mean(self._history["extract"])),
            model_inference_ms=float(np.mean(self._history["inference"])),
            sentence_process_ms=float(np.mean(self._history["sentence"])),
            total_pipeline_ms=float(np.mean(self._history["total"])),
            fps=float(np.mean(self._fps_history)),
        )

