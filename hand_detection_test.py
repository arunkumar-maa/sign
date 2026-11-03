
import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
from pathlib import Path

# --------------------------------
# 1. Download MediaPipe hand model
# --------------------------------

model_folder = Path("models")
model_folder.mkdir(exist_ok=True)

model_path = model_folder / "hand_landmarker.task"

model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

if not model_path.exists():
    print("Downloading hand detection model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded successfully!")

# --------------------------------
# 2. Configure MediaPipe
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

# --------------------------------
# 3. Hand connections
# --------------------------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# --------------------------------
# 4. Start webcam
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera not detected!")
    raise SystemExit

start_time = time.perf_counter()
last_timestamp = -1

print("Hand detection started. Press Q to exit.")

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Cannot read camera frame")
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            rgb_frame = np.ascontiguousarray(rgb_frame)

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Unique increasing video timestamp
            timestamp = max(
                int((time.perf_counter() - start_time) * 1000),
                last_timestamp + 1
            )

            last_timestamp = timestamp

            # Detect hands
            result = detector.detect_for_video(
                mp_image, timestamp
            )

            hand_count = len(result.hand_landmarks)

            if hand_count == 0:
                cv2.putText(
                    frame,
                    "NO HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Show your open palm clearly",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # Draw detected hand landmarks
            for hand_index, hand in enumerate(
                result.hand_landmarks
            ):

                points = []

                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))

                    points.append((x, y))

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw finger connections
                for start, end in connections:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 0),
                        2
                    )

                cv2.putText(
                    frame,
                    "21 LANDMARKS DETECTED",
                    (20, 120 + hand_index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Hands detected: {hand_count}",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("SignBridge AI - Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
