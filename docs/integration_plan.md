# SignBridge AI — Live ISL Hand Detection and Sign Recognition Integration Plan

**Date:** 2026-11-03  
**Status:** Approved for Implementation  
**Lead Engineer:** Senior Computer Vision & ISL ML Engineer  

---

## 1. Project Inspection & Architecture Assessment

### 1.1 Existing Application Structure
* **Application Entry Point:** `app.py`
  - Dual launch: Background FastAPI service on port 8000 + Gradio Blocks interface on port 7860.
* **Current UI Modules (`frontend/`):**
  - `home.py`: Welcome tab.
  - `diagnostics_ui.py`: Standalone camera telemetry and 9-point posture testing.
  - `translator_ui.py`: Core live translation tab.
  - `communication_ui.py`: Two-way conversation interface.
  - `learning_ui.py`: ISL flashcards and sign training.
  - `history_ui.py`: Saved conversation logs.
  - `about_ui.py`: Project documentation and credentials.
* **Current Webcam Pipeline:**
  - Gradio `gr.Image(sources="webcam", type="numpy", streaming=True)` in `frontend/translator_ui.py`.
  - Color handling: Gradio supplies frames as **RGB** numpy arrays.
* **Separate Hand Detection Implementation (`hand_detection_test.py`):**
  - Uses `mediapipe.tasks.vision.HandLandmarker` with `RunningMode.VIDEO` and `models/hand_landmarker.task`.
  - Renders **green landmark circles** `(0, 255, 0)` with radius 5.
  - Renders **blue finger connection lines** `(255, 0, 0)` with thickness 2.
  - Displays HUD status: "21 LANDMARKS DETECTED", "Hands detected: {count}", "NO HAND DETECTED", "Show your open palm clearly".
  - Uses `cv2.imshow()` and `cv2.VideoCapture()`, which creates a standalone desktop window.
  - **Objective:** Move this exact detection, drawing, and status logic directly inside the existing Gradio webcam stream of `frontend/translator_ui.py` without opening external windows.

---

## 2. Target File Modifications & Creations

| Module / Component | Target File | Action | Purpose |
|---|---|:---:|---|
| **Hand Landmarker** | `backend/vision/hand_landmarks.py` | **Update** | Support detection thresholds (0.3), green dots (radius 5), blue lines, exact connections, single reusable instance in IMAGE/VIDEO mode. |
| **Landmark Normalization** | `backend/vision/landmark_preprocessing.py` | **Create** | Consistent normalization: relative to wrist/bounding region, scale invariant, one-hand/two-hand support, hand-presence masks. |
| **Dataset Download** | `ml/dataset_download.py` | **Create** | Download & verify `vidit031/isl-isolated-40words` from Hugging Face Hub (642 clips, 40 classes). |
| **Dataset Audit** | `ml/dataset_audit.py` | **Create** | Verify metadata CSV, video paths, class labels, provenance, licenses, and detect corrupted clips. |
| **Landmark Extraction** | `ml/extract_landmarks.py` | **Create** | Extract 21 hand landmarks (per hand) from video frames with timestamps and signer IDs. |
| **Sequence Builder** | `ml/build_sequences.py` | **Create** | Uniformly sample/interpolate fixed-length sequences (30 frames) with normalization and train/val/test splits partitioned by signer. |
| **Model Training** | `ml/train_word_model.py` | **Create** | Train Bidirectional LSTM/GRU sequence classifier on normalized ISL video sequences; save `models/isl_word_model.keras` & `models/word_labels.json`. |
| **Model Evaluation** | `ml/evaluate.py` | **Update** | Compute accuracy, precision, recall, F1, confusion matrix, per-class metrics, and inference latency. |
| **ISL Word Recognizer** | `backend/recognition/isl_word_recognizer.py` | **Create** | Thread-safe sliding window buffer (30 frames), unknown-sign rejection, prediction smoothing, cooldown logic. |
| **ISL Letter Recognizer** | `backend/recognition/isl_letter_recognizer.py` | **Create** | Dedicated ISL fingerspelling/alphabet recognizer (isolated from word model), with clear "Training data required" status when uncalibrated. |
| **Sentence Builder** | `backend/recognition/sentence_builder.py` | **Update** | Support Add Word, Undo, Clear Sentence, Reset, duplicate prevention, and ordered sentence display. |
| **Text to Speech** | `backend/audio/text_to_speech.py` | **Preserve/Update** | Speak sentence in English (Edge-TTS / gTTS fallback) without blocking the camera thread. |
| **Unified Single Screen** | `frontend/translator_ui.py` | **Update** | Integrate camera feed + green/blue skeleton + telemetry + Word/Letter mode + predictions + sentence controls + TTS into one responsive screen. |
| **Automated Tests** | `tests/test_isl_integration.py` | **Create** | Pytest coverage for camera processing, hand landmarker, normalization, model inference, unknown signs, sentence builder, TTS. |
| **Implementation Status** | `docs/implementation_status.md` | **Update** | Document final progress, evaluation metrics, and run instructions. |

---

## 3. Step-by-Step Implementation Sequence

1. **Phase 2 & 3:** Upgrade `backend/vision/hand_landmarks.py` to match the exact detection parameters (0.3 threshold, green circles radius 5, blue connection lines, 2 hands, 21 landmarks, FPS, status messages).
2. **Phase 4:** Build `backend/vision/landmark_preprocessing.py` providing uniform scale/wrist normalization, handedness ordering (Left/Right), and presence masking.
3. **Phase 5 & 6:** Verify/extract dataset via `ml/dataset_download.py`, `ml/dataset_audit.py`, `ml/extract_landmarks.py`, `ml/build_sequences.py`, train the sequence model via `ml/train_word_model.py`, and save `models/isl_word_model.keras` + `models/word_labels.json`.
4. **Phase 7 & 8:** Implement `backend/recognition/isl_word_recognizer.py` and `backend/recognition/isl_letter_recognizer.py`.
5. **Phase 9:** Connect `backend/recognition/sentence_builder.py`.
6. **Phase 10 & 11:** Upgrade `frontend/translator_ui.py` to host all controls on one screen with TTS.
7. **Phase 12, 13 & 14:** Execute test suite and update documentation.

