# SignBridge AI – Camera and Vision Diagnostic Report

**Date:** 2026-11-03  
**Status:** Complete Forensic Inspection (Phase 1)  
**Author:** SignBridge AI Engineering Team  

---

## 1. Project Inspection Summary

The SignBridge AI repository is structured as a modular Python-only system for Indian Sign Language (ISL) recognition:
- **`backend/`**:
  - `backend/camera/frame_processor.py`: Orchestrates multi-modal tracking and frame annotation.
  - `backend/vision/`: Landmark extraction modules (`hand_landmarks.py`, `face_landmarks.py`, `pose_landmarks.py`, `feature_extraction.py`).
  - `backend/recognition/`: Static (`static_recognizer.py`), dynamic (`dynamic_recognizer.py`), temporal filtering (`temporal_filter.py`), and sentence generation (`sentence_builder.py`).
  - `backend/translation/translator.py`: Multi-language translation with offline dictionary fallback.
  - `backend/audio/`: TTS (`text_to_speech.py`) and STT (`speech_to_text.py`).
  - `backend/database/`: SQLite database client for conversation logs (`signbridge.db`).
- **`frontend/`**: Gradio tabs for Live Translation (`translator_ui.py`), Two-way communication (`communication_ui.py`), Learning (`learning_ui.py`), History (`history_ui.py`), and Emergency (`emergency_ui.py`).
- **`ml/`**: Preprocessing (`preprocess_isl40.py`, `preprocessing.py`), training (`train_isl40_dynamic.py`, `train_static.py`), evaluation (`evaluate.py`), and dataset utilities.
- **`models/`**: Keras models, label dictionaries, metadata reports, and MediaPipe storage.
- **`data/`**: Raw video clips (`data/raw/isl_isolated_40words/`), processed landmark arrays (`data/processed/isl40/`), and SQLite database.
- **`tests/`**: Unit test suite (`tests/test_core.py`).

---

## 2. Existing Camera Implementation

- **Interface:** Gradio `gr.Image(sources=["webcam"], streaming=True, type="numpy", mirror_webcam=True)` inside `frontend/translator_ui.py`.
- **Callback:** `process_frame` streams browser-captured frames into the backend `FrameProcessor`.
- **Deficiency:** There was no standalone OpenCV hardware verification script (`tests/manual_camera_test.py` was absent), making it impossible to debug local webcam device indices, real driver FPS, capture resolution, and hardware permissions independently of browser WebRTC.

---

## 3. MediaPipe Version Analysis

- **Installed Version:** `mediapipe==0.10.18` (Python 3.11.9 on Windows x64).
- **Architecture Used:** Legacy `mp.solutions.hands`, `mp.solutions.face_mesh`, `mp.solutions.pose`.
- **Status:** Google officially deprecated `mp.solutions` in MediaPipe 0.10.x in favor of the **MediaPipe Tasks API** (`mediapipe.tasks.python.vision`). The legacy solutions in 0.10.18 suffer from synchronization quirks and lack the latest BlazeHand V2 / HandLandmarker features.

---

## 4. Installed Dependencies Audit

Verification against requirements:
| Dependency | Version Detected | Status |
|---|---|---|
| `python` | 3.11.9 | Supported |
| `mediapipe` | 0.10.18 | Supported (Tasks API available) |
| `opencv-python` | 4.10.0.84 | Supported |
| `tensorflow` | 2.16.1 | Supported |
| `keras` | 3.15.1 | Supported |
| `gradio` | 4.44.1 | Supported |
| `fastapi` | 0.115.0 | Supported |
| `uvicorn` | 0.30.6 | Supported |
| `numpy` | 1.26.4 | Supported |
| `pandas` | 2.2.3 | Supported |
| `scikit-learn` | 1.5.2 | Supported |
| `pytest` | 8.3.3 | Supported |
| `huggingface-hub` | 0.25.2 | Supported |
| `gTTS` | 2.5.4 | Supported |
| `faster-whisper` | 1.2.1 | Supported |

---

## 5 & 6. Existing Model Files & Provenance Verification

### Dynamic Models:
- `models/dynamic/isl_dynamic_40words.keras` (2.69 MB)
  - **Status:** **Real trained model.**
  - **Dataset:** `vidit031/isl-isolated-40words` (642 raw videos across 40 ISL vocabulary classes).
  - **Architecture:** Bidirectional LSTM with Spatial Dropout and Batch Normalization.
  - **Performance:** Test accuracy ~45.8% (evaluated on held-out pure test splits).
- `models/dynamic/isl_dynamic_v1.keras` (2.69 MB)
  - Duplicate copy of the 40-word dynamic model.

### Static Models:
- `models/static/isl_static_v1.keras` (3.27 MB)
  - **Status:** **Synthetic / Mock trained.**
  - **Issue:** Generated via `ml/generate_and_train_all.py` using synthetic rule-based coordinate offsets (`create_hand_pattern()`) rather than real captured camera frames or actual ISL signers.
  - **Labels:** 39 classes (A-Z, 1-5, and 8 macro signs).

### MediaPipe Models:
- `models/mediapipe/`: **Completely empty.**
- **Missing Required Tasks Assets:**
  - `hand_landmarker.task`
  - `face_landmarker.task`
  - `pose_landmarker.task`

---

## 7 & 8. Camera Callback & RGB/BGR Inversion Forensic Analysis

### The Critical Color-Inversion Bug:
1. **Gradio Delivery:** `gr.Image(sources=["webcam"], type="numpy")` delivers image arrays in **RGB** color space.
2. **Backend Assumption:** `backend/camera/frame_processor.py` and `backend/vision/hand_landmarks.py` assumed input was OpenCV **BGR**:
   ```python
   # Inside backend/vision/hand_landmarks.py
   rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
   ```
3. **Double Inversion:** Passing an RGB array to `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` swaps Red and Blue channels, producing a **BGR** array!
4. **Detector Failure:** The MediaPipe palm detection neural network was fed inverted skin tones (blue hands). MediaPipe's single-shot detector is trained on natural RGB skin reflectance spectra; channel inversion causes palm detection confidence to plummet below the default 0.5 threshold, resulting in complete detection dropouts.
5. **Visualization Inversion:** `frontend/translator_ui.py` performed an additional `cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)` before returning to Gradio, corrupting visual overlay colors.

---

## 9. MediaPipe Model Asset Paths

- The current implementation invoked `mp.solutions.hands.Hands()`, which looks for legacy C++ embedded graphs.
- To meet the strict requirements of Phase 3, 4, and 5, the project must migrate to `mediapipe.tasks.python.vision.HandLandmarker` loaded from `models/mediapipe/hand_landmarker.task`.
- No automatic download or fallback logic existed for missing `.task` bundle files.

---

## 10. Root Causes of Landmark Failure & Test Breakages

1. **RGB/BGR Channel Swapping:** The primary cause of failure in the Gradio webcam interface was color inversion before feeding to MediaPipe.
2. **Missing Tasks Model Bundle:** Absence of `hand_landmarker.task` prevented using Google's official Edge Hand Landmarker API.
3. **Ayzek / ASL Engine Contamination:** `backend/recognition/static_recognizer.py` imported `AyzekRecognizer` which loaded American Sign Language (ASL) rules from an unverified external path, causing circular import failures and breaking `test_static_recognizer_unavailable_message` because `is_loaded` reported `True` even when no model existed.
4. **Lack of Hardware Diagnostics:** Absence of a dedicated camera test tool prevented distinguishing between hardware/OS webcam permission issues, lighting deficiencies, and pipeline errors.

---

## Next Immediate Steps (Phase 2 & 3)

1. **Phase 2:** Build standalone OpenCV test `tests/manual_camera_test.py` to verify camera hardware, resolution, and frame rates.
2. **Phase 3:** Download `hand_landmarker.task` and implement `backend/vision/hand_landmarks.py` using `mediapipe.tasks.python.vision.HandLandmarker`, supporting both OpenCV BGR and Gradio RGB inputs with robust error handling.

