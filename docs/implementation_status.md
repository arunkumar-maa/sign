# SignBridge AI — Implementation & Diagnostic Status

**Last Updated:** Final GitHub Audit Corrections Complete  
**Tagline:** *Breaking Communication Barriers with AI (Camera-First Indian Sign Language Recognition)*  
**Architecture:** Python-Only (Gradio Blocks + FastAPI + MediaPipe Tasks API + TensorFlow/Keras + NumPy/Pandas + Faster-Whisper + Edge-TTS + SQLite)  
**Primary Focus:** Exclusive Real-Time Recognition of ISL English Alphabet (A–Z, 26 classes) & Numbers (0–9, 10 classes) with Character-by-Character Text Generation.

---

## 1. Active Implementation Summary

| Component | Target Scope | Active Status | Implementation File |
|---|---|:---:|---|
| **ISL Alphabet Recognition** | 26 classes (A–Z) | **ACTIVE & VERIFIED** | `backend/recognition/isl_fingerspelling_recognizer.py` |
| **ISL Number Recognition** | 10 classes (0–9) | **ACTIVE & VERIFIED** | `backend/recognition/isl_number_recognizer.py` |
| **Character Text Builder** | Character-by-character (`HELLO123`) | **ACTIVE & VERIFIED** | `backend/recognition/character_builder.py` |
| **Unified Feature Schema** | 282-dim invariant vectors | **ACTIVE & VERIFIED** | `backend/vision/feature_schema.py` |
| **Live Camera Pipeline** | Single annotated stream (green dots + blue skeleton) | **ACTIVE & VERIFIED** | `backend/vision/live_pipeline.py` |
| **User Interface** | Alphabet & Numbers mode toggle, session-isolated | **ACTIVE & VERIFIED** | `frontend/translator_ui.py` |
| **Asset Setup & Verification** | MediaPipe Task & BiLSTM checks | **ACTIVE & VERIFIED** | `scripts/setup_assets.py` |
| **Signer-Independent Training** | >= 3 signers, zero random split fallback | **ACTIVE & VERIFIED** | `scripts/train_alphabet.py`, `scripts/train_numbers.py` |

---

## 2. Phase-by-Phase Execution Progress Tracker

| Phase | Description | Status | Evidence / Verification |
|---|---|:---:|---|
| **Phase 1: Diagnostic Report** | Investigated false word predictions on two fingers (`HELLO`, `GO`, `STUDENT`); analyzed softmax forcing and lack of motion gating. | **COMPLETED** | `docs/recognition_debug_report.md` |
| **Phase 2: Invariant Finger Counter** | Built 3D rotation-invariant finger counter (0-5) using PIP/DIP joint angles and palm-scale distance ratios. | **COMPLETED** | `backend/recognition/finger_counter.py` |
| **Phase 3: ISL Number Recognizer** | Dedicated ISL number classifier (0-9) adhering to official ISL syntax and rules. | **COMPLETED** | `backend/recognition/isl_number_recognizer.py` |
| **Phase 4: Dynamic Word Recognizer** | *[HISTORICAL / ARCHIVED]* 40-word model with motion energy gating. | **ARCHIVED** | Retained in `backend/recognition/isl_word_recognizer.py` for reference. Decoupled from active pipeline. |
| **Phase 5: Mode Decoupling** | Exposes strictly `Alphabet (A–Z)` and `Numbers (0–9)`. | **COMPLETED** | `LivePipeline.set_mode()`, UI radio toggle. |
| **Phase 6: Character Text Builder** | Character-by-character assembly (`HELLO123`) with manual add, backspace, clear, and auto-add neutral-release gating. | **COMPLETED** | `backend/recognition/character_builder.py` |
| **Phase 7: Single Camera Display** | Exactly ONE camera output showing video with green landmark dots and blue skeleton connections. | **COMPLETED** | `frontend/translator_ui.py` |
| **Phase 8: Latency Profiler** | Downscaled MediaPipe detection to 320x240 for sub-35ms detection latency. | **COMPLETED** | `backend/vision/performance_monitor.py` |
| **Phase 9: Pipeline & UI Integration** | Integrated character builder, session isolation, multilingual TTS, and translations. | **COMPLETED** | `backend/vision/live_pipeline.py` & `frontend/translator_ui.py` |
| **Phase 10: Test Verification** | 100% test pass rate across all verification suites. | **COMPLETED** | `tests/test_alphabet_and_numbers_suite.py`, `tests/test_live_pipeline_repaired.py`, etc. |

---

## 3. Key Errors Discovered & Resolved

1. **Softmax Forcing on Stationary Handshapes:**
   - *Discovery:* Closed 40-class softmax forced 100% probability across vocabulary words even for stationary, out-of-distribution handshapes (e.g. 2 fingers $\to$ `TEA: 81.6%`, `HELLO: 78.4%`).
   - *Fix:* Refocused application strictly on Alphabet (A–Z) and Numbers (0–9). Removed word recognition from the active pipeline and enforced strict confidence thresholds ($> 0.50$ / $0.65$). Static or unconfident hands return `Unknown Sign`.

2. **Duplicate Camera Stream in Interface:**
   - *Discovery:* Gradio displayed raw webcam input and annotated stream side-by-side.
   - *Fix:* Unified interface to display exactly ONE camera image (`camera_annotated`) rendering green dots and blue skeleton connections. Raw hardware capture is neatly collapsed inside a device settings accordion.

3. **Continuous Character Duplication in Text Builder:**
   - *Discovery:* Auto-add added identical characters repeatedly while user held hand steady (e.g. `AAAAA`).
   - *Fix:* Defaulted to manual confirmation (`auto_add_enabled = False`); implemented a 6-consecutive-frame stability gate (>0.65 confidence) followed by a mandatory neutral-hand release before allowing repetition of the same character.

4. **Training Split Contamination:**
   - *Discovery:* Previous scripts used random `train_test_split` fallbacks that mixed clips from the same signer across train and test sets.
   - *Fix:* Enforced genuine signer-independent splits (train, validation, test) requiring $\ge 3$ distinct signers in `scripts/train_alphabet.py` and `scripts/train_numbers.py`. Prohibited random splits entirely.

5. **Number Class Standardization:**
   - *Discovery:* Mixed representations (`TWO` vs `2`) caused confusion in training manifests.
   - *Fix:* Standardized number model output to exactly 10 digits (`0` through `9`). Handled word conversions strictly during dataset validation and import in `isl_data_collection/storage.py`.
