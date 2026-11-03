"""
SignBridge AI — Manual & Standalone OpenCV Camera Test.
Verifies camera capture, frame delivery, resolution, FPS, and clean teardown.
Can run interactively with GUI or headlessly for automated diagnostics.
"""

import argparse
import sys
import time
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Exclude from automatic pytest collection (manual interactive test tool)
__test__ = False


def test_camera(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    max_frames: int = 0,
    headless: bool = False,
) -> bool:
    """
    Open webcam, capture frames, measure FPS, and cleanly close.

    Args:
        camera_index: OpenCV camera device ID (0 is standard default).
        width: Requested horizontal frame resolution.
        height: Requested vertical frame resolution.
        target_fps: Target camera capture rate.
        max_frames: Number of frames to test (0 for continuous interactive mode).
        headless: If True, skips cv2.imshow for CI/automated environments.

    Returns:
        bool: True if camera successfully captured valid frames, False otherwise.
    """
    logger.info("Initializing camera device index %d...", camera_index)

    # On Windows, DirectShow backend (CAP_DSHOW) initializes much faster than MSMF
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("CAP_DSHOW failed, falling back to standard backend...")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error("Failed to open camera with index %d. No camera hardware detected or access denied.", camera_index)
        return False

    # Configure resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info("Camera successfully opened.")
    logger.info("Resolution: requested %dx%d -> actual %dx%d", width, height, actual_w, actual_h)
    logger.info("Reported hardware FPS: %.1f", reported_fps)

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_display = 0.0
    success = False

    window_name = "SignBridge AI - Camera Diagnostic (Press 'q' or ESC to exit)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.error("Failed to read frame from camera at frame #%d.", frame_count)
                break

            frame_count += 1
            success = True

            # Calculate FPS every 10 frames
            curr_time = time.time()
            if frame_count % 10 == 0:
                fps_display = 10.0 / (curr_time - last_fps_time) if (curr_time - last_fps_time) > 0 else 0.0
                last_fps_time = curr_time

            # If running in GUI mode, render HUD and display
            if not headless:
                hud_frame = frame.copy()
                hud_text = f"Frames: {frame_count} | Res: {actual_w}x{actual_h} | FPS: {fps_display:.1f}"
                cv2.rectangle(hud_frame, (10, 10), (450, 50), (0, 0, 0), -1)
                cv2.putText(
                    hud_frame,
                    hud_text,
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    hud_frame,
                    "Press 'q' to exit",
                    (20, actual_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                try:
                    cv2.imshow(window_name, hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        logger.info("User requested exit.")
                        break
                except cv2.error:
                    logger.warning("cv2.imshow unavailable in current display environment; switching to headless.")
                    headless = True

            if max_frames > 0 and frame_count >= max_frames:
                logger.info("Reached maximum target frames (%d). Exiting loop.", max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0.0
        logger.info("Closing camera resources...")
        cap.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("Camera closed cleanly.")
        logger.info("Summary: %d frames captured in %.2f seconds (Avg FPS: %.2f)", frame_count, total_time, avg_fps)

    return success and (frame_count > 0)


def main():
    parser = argparse.ArgumentParser(description="Standalone OpenCV Camera Verification Tool")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--frames", type=int, default=0, help="Frames to capture (0 for continuous)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    ok = test_camera(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        max_frames=args.frames,
        headless=args.headless,
    )
    if ok:
        logger.info("SUCCESS: Camera verification passed.")
        sys.exit(0)
    else:
        logger.warning("WARNING/FAILURE: Camera could not be verified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
