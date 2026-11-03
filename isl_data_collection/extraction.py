import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from .config import FEATURE_DIR, LANDMARK_MANIFEST, SAMPLES_MANIFEST
from .hand_landmarks import HandPoseDetector, result_to_features
from .storage import append_landmark_record, read_jsonl


def extract_one_video(record: dict, frame_stride: int = 2) -> tuple[np.ndarray, int]:
    video_path = Path(record["video_path"])
    if not video_path.is_absolute():
        video_path = SAMPLES_MANIFEST.parent.parent / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 120:
        fps = float(record.get("fps", 10.0)) or 10.0

    vectors = []
    frames_with_hand = 0
    frame_index = 0
    detector = HandPoseDetector(running_mode=mp.tasks.vision.RunningMode.VIDEO)
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index % max(1, frame_stride) == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(0, int(frame_index * 1000.0 / fps))
                result = detector.detect_video(rgb, timestamp_ms)
                features, hand_count = result_to_features(result)
                vectors.append(features)
                if hand_count:
                    frames_with_hand += 1
            frame_index += 1
    finally:
        capture.release()
        detector.close()

    if len(vectors) < 5:
        raise RuntimeError("Too few frames were extracted. Record a longer sign clip.")
    if frames_with_hand < max(3, int(len(vectors) * 0.25)):
        raise RuntimeError("Hands were detected in too few frames. Check hand visibility and lighting.")
    return np.stack(vectors).astype(np.float32), frames_with_hand


def extract_dataset(force: bool = False, frame_stride: int = 2) -> str:
    records = read_jsonl(SAMPLES_MANIFEST)
    if not records:
        return "No recordings found. Record and save labelled clips first."

    # Rebuild the processed manifest so removed/re-recorded source samples cannot leave stale entries.
    LANDMARK_MANIFEST.unlink(missing_ok=True)
    processed, skipped = 0, 0
    messages = []
    for index, record in enumerate(records, start=1):
        sample_id = record.get("sample_id", f"sample_{index}")
        out_path = FEATURE_DIR / f"{sample_id}.npz"
        if out_path.exists() and not force:
            try:
                existing = np.load(out_path)["features"]
                if existing.ndim == 2 and existing.shape[1] == 282:
                    saved = dict(record)
                    saved["features_path"] = str(out_path.relative_to(FEATURE_DIR.parent.parent))
                    saved["extracted_frames"] = int(existing.shape[0])
                    append_landmark_record(saved)
                    processed += 1
                    continue
            except Exception:
                pass
        try:
            features, frames_with_hand = extract_one_video(record, frame_stride=frame_stride)
            np.savez_compressed(out_path, features=features)
            saved = dict(record)
            saved["features_path"] = str(out_path.relative_to(FEATURE_DIR.parent.parent))
            saved["extracted_frames"] = int(features.shape[0])
            saved["frames_with_hand"] = int(frames_with_hand)
            append_landmark_record(saved)
            processed += 1
            messages.append(f"OK {sample_id}: {features.shape[0]} frames")
        except Exception as exc:
            skipped += 1
            messages.append(f"SKIP {sample_id}: {exc}")

    summary = f"Landmark extraction complete. Processed: {processed}; skipped: {skipped}."
    return summary + ("\n" + "\n".join(messages[-20:]) if messages else "")
