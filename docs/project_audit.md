# SignBridge AI — Comprehensive Project Audit & Architectural Repair Report

**Date:** 2026-11-04  
**Author:** SignBridge AI Engineering Team  
**Git Branch:** `main`  
**Repository:** https://github.com/arunkumar-maa/sign.git  
**Target Focus:** Exclusive Real-Time Recognition of ISL English Alphabet (A–Z, 26 classes) & Numbers (0–9, 10 classes) with Character-by-Character Text Generation.

---

## 1. Executive Summary

An exhaustive end-to-end audit of the **SignBridge AI** repository was performed covering the entire architecture, machine learning pipelines, feature extraction modules, camera feeds, UI flows, backend configuration, and test suites.

The primary defect previously reported by users—where showing a two-finger hand gesture produced arbitrary ISL word predictions such as "HELLO", "GO", "SORRY", or "STUDENT"—was traced to **three structural root causes**:
1. **Feature Vector Dimension Inconsistency and Silent Padding/Slicing**: Multiple conflicting feature extraction implementations existed across the repository (104-dim, 126-dim, 192-dim, 216-dim, and 282-dim). When a model expected 216 or 192 dimensions, recognizers silently sliced or zero-padded the vector (`seq_arr[:, :expected_dim]`), destroying topological landmark relationships and feeding nonsensical vectors into dense and LSTM layers.
2. **Scope Dilution Towards Large-Vocabulary Words**: The application was attempting word-level continuous sign translation and gloss conversion instead of the core requirement: **accurate ISL alphabet fingerspelling (A–Z) and numbers (0–9)**.
3. **Absence of Shared Versioned Feature Schema**: Inference code in `live_pipeline.py`, collection code in `isl_data_collection/`, and training scripts each extracted landmarks with subtly different normalizations, angles, and wrist scales.

This audit details the problems, root causes, affected files, proposed architectural solutions, and the implementation status for all 15 audit dimensions.

---

## 2. Comprehensive Issue Audit Matrix

| ID | Issue Category | Problem Description | Root Cause | Affected Files | Proposed Solution | Status |
|---|---|---|---|---|---|---|
| **AUD-01** | **Feature Schema** | Mismatched feature dimensions across modules (104, 192, 216, 282). | Multiple disconnected feature extractors without a single source of truth. | `ml/preprocessing.py`<br>`backend/vision/hand_pose_features.py`<br>`isl_data_collection/hand_landmarks.py`<br>`backend/vision/live_pipeline.py` | Create versioned `backend/vision/feature_schema.py` (v2.0.0, 282 dims). Strict validation; raise `IncompatibleModelError` instead of padding/slicing. | **FIXED** |
| **AUD-02** | **Feature Compatibility** | Discrepancies between training feature extraction and live webcam inference. | Training scripts and live webcam used different angle normalization and scaling logic. | `scripts/import_fingerspelling.py`<br>`scripts/import_number_datasets.py`<br>`backend/vision/live_pipeline.py` | Ensure training and live inference call identical extraction functions from `backend.vision.feature_schema`. | **FIXED** |
| **AUD-03** | **Model Slicing / Padding** | Silent zero-padding or vector truncation when shapes mismatch. | Defensive try/except blocks in recognizers truncated or padded arrays to prevent runtime exceptions. | `backend/recognition/isl_fingerspelling_recognizer.py`<br>`backend/recognition/isl_number_recognizer.py`<br>`backend/recognition/isl_word_recognizer.py` | Forbid silent reshape/slicing. Validate input shape `(None, 30, 282)` on model load and frame feed. Reject mismatched models explicitly. | **FIXED** |
| **AUD-04** | **Scope Creep (Words)** | Word recognition, gloss sequences, and sentence translation cluttering core objective. | System previously attempted 40-word ISL gloss translation, distracting from Alphabet and Numbers. | `frontend/translator_ui.py`<br>`backend/vision/live_pipeline.py`<br>`backend/recognition/sentence_builder.py` | Refactor application to focus exclusively on Alphabet (A–Z) and Numbers (0–9). De-emphasize word recognition. | **FIXED** |
| **AUD-05** | **Wrong Predictions** | Arbitrary predictions on static finger poses (e.g. 2 fingers $\rightarrow$ "HELLO"). | Model was forced to choose argmax from word models with low threshold; absence of confidence gating. | `backend/recognition/` recognizers | Implement explicit confidence thresholding ($> 0.65$). Return `"Unknown"` when unconfident or hand absent. | **FIXED** |
| **AUD-06** | **Character Builder** | Sentence builder treated characters as whole words separated by spaces. | `SentenceBuilder` was designed for word glosses (`"HELLO" + " " + "WORLD"`). | `backend/recognition/sentence_builder.py`<br>`frontend/translator_ui.py` | Create `CharacterTextBuilder` designed for character accumulation (e.g. `HELLO123`), with Add, Backspace, Clear, and TTS. | **FIXED** |
| **AUD-07** | **Auto-Add Runaway** | Characters repeatedly appended without user release, producing runs like `AAAAA`. | Auto-add lacked required neutral-hand release or transition cooldown. | `backend/vision/live_pipeline.py`<br>`frontend/translator_ui.py` | Implement stability requirement (N consecutive frames) followed by a mandatory neutral release before next character. | **FIXED** |
| **AUD-08** | **Session Isolation** | Global mutable singletons shared across users; text leaked between concurrent browser tabs. | `_pipeline = LivePipeline()` instantiated at module level in `translator_ui.py`. | `frontend/translator_ui.py` | Use Gradio `gr.State` per user session to hold pipeline and character text builder state. | **FIXED** |
| **AUD-09** | **Duplicate Service Init** | Heavy services initialized twice on startup, slowing down launch. | `_init_services()` called synchronously in `app.py` and again in FastAPI `@app.on_event("startup")`. | `app.py`<br>`backend/main.py` | Use FastAPI `lifespan` handler; eliminate redundant pre-startup initialization call. | **FIXED** |
| **AUD-10** | **CORS & Security** | Overly permissive CORS with `allow_origins=["*"]` combined with `allow_credentials=True`. | Non-standard Starlette configuration causing browser security warnings. | `backend/main.py`<br>`config.py` | Restrict default CORS origins to local development hosts and allow explicit configuration via `.env`. | **FIXED** |
| **AUD-11** | **Camera Display** | Redundant or confusing dual camera feeds in some UI sections. | Raw camera and landmark overlay separated in UI layouts. | `frontend/translator_ui.py`<br>`isl_data_collection/ui.py` | Exactly one primary camera feed displaying video with green landmark dots and blue connections directly overlaid. | **FIXED** |
| **AUD-12** | **Alphabet Pipeline** | Alphabet recognition lacked explicit 26-class mapping (A–Z) and static vs temporal handling. | Static letters and dynamic letters (J, Z) treated identically with arbitrary temporal resampling. | `backend/recognition/isl_fingerspelling_recognizer.py` | Explicit A–Z label map, motion energy tracking for dynamic letters (J, Z), static handshape evaluation for static letters. | **FIXED** |
| **AUD-13** | **Number Pipeline** | Confusion between geometric finger counter (0–5) and full ISL number recognition (0–9). | Two different modules handled numbers with conflicting outputs. | `backend/recognition/isl_number_recognizer.py`<br>`backend/recognition/finger_counter.py` | Standardize on `ISLNumberRecognizer` (0–9) using the 282-dim trained model, keeping finger counter as diagnostic only. | **FIXED** |
| **AUD-14** | **Dataset Collection** | Collection interface allowed generic word recording, missing Alphabet/Number validation. | Collection UI was oriented towards words without landmark quality checks. | `isl_data_collection/ui.py`<br>`isl_data_collection/recording.py` | Restrict collection to Alphabet (A–Z) and Numbers (0–9); validate that at least one hand is detected before saving. | **FIXED** |
| **AUD-15** | **Training Scripts** | Missing unified standalone training scripts for Alphabet and Numbers using the shared schema. | Ad-hoc scripts scattered across `ml/` and `scripts/`. | `scripts/train_alphabet.py`<br>`scripts/train_numbers.py` | Create dedicated, clean training scripts with held-out signer evaluation and model export. | **FIXED** |

---

## 3. Deep-Dive Analysis of Core Components

### 3.1 Feature Extraction Schema (282 Dimensions)
The unified feature vector represents two hands in 282 floating-point values:
- **Left Hand Slot (140 floats)**:
  - 63 values: 21 landmarks $\times$ 3 raw 3D coordinates $(x, y, z)$.
  - 63 values: 21 landmarks $\times$ 3 wrist-relative coordinates $(p - \text{wrist}) / \text{palm\_scale}$.
  - 10 values: Joint angles normalized to $[0, 1]$ (Thumb: CMC-MCP-IP, MCP-IP-TIP; Fingers: MCP-PIP-DIP, PIP-DIP-TIP).
  - 4 values: Normalized fingertip-to-thumb distances (Index, Middle, Ring, Pinky to Thumb Tip).
- **Right Hand Slot (140 floats)**:
  - Identical 140-value formulation for the right hand.
- **Hand Presence Flags (2 floats)**:
  - `[is_left_present, is_right_present]` ($1.0$ or $0.0$).
- **Total Dimensions**: $140 + 140 + 2 = 282$.
- **Sequence Length**: 30 frames ($30 \times 282 = 8,460$ values per temporal sequence).

### 3.2 Machine Learning Architecture
- **Alphabet Recognizer (`models/isl_fingerspelling_model.keras`)**:
  - Input: `(None, 30, 282)`
  - Architecture: Bidirectional LSTM (64 units) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (64, ReLU) $\rightarrow$ Dense (26, Softmax).
  - Output: 26 classes corresponding strictly to English letters `A` through `Z`.
- **Number Recognizer (`models/isl_number_model.keras`)**:
  - Input: `(None, 30, 282)`
  - Architecture: Bidirectional LSTM (64 units) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (64, ReLU) $\rightarrow$ Dense (10, Softmax).
  - Output: 10 classes corresponding strictly to Digits `0` through `9`.

### 3.3 Character Text Builder Specification
The previous word-based sentence assembler is replaced with a **Character Text Builder**:
- Assembles readable strings character-by-character (e.g., `HELLO123`).
- **Add Character**: Appends current confident prediction to text buffer.
- **Backspace**: Removes the trailing character.
- **Clear Text**: Resets the text buffer.
- **Speak Text**: Triggers local Python Text-to-Speech (Edge-TTS or gTTS fallback).
- **Auto-Add Guard**: Requires hand stability for 6+ frames ($>0.65$ confidence) followed by a neutral-release gap before re-triggering.

---

## 4. Verification and Implementation Checklist

- [x] Create separate repair branch `repair/alphabet-numbers-refactor` and merge into `main`.
- [x] Perform full repository inventory and audit report.
- [x] Implement `backend/vision/feature_schema.py` (v2.0.0, 282-dim, validation, compatibility checks).
- [x] Refactor Alphabet Recognizer (`backend/recognition/isl_fingerspelling_recognizer.py`).
- [x] Refactor Number Recognizer (`backend/recognition/isl_number_recognizer.py`).
- [x] Refactor Live Pipeline (`backend/vision/live_pipeline.py`) with Character Text Builder.
- [x] Overhaul Dataset Collection UI (`isl_data_collection/ui.py`).
- [x] Create dedicated training scripts: `scripts/train_alphabet.py` and `scripts/train_numbers.py`.
- [x] Refactor Live Webcam UI (`frontend/translator_ui.py`) with per-session state isolation.
- [x] Clean backend initialization in `app.py` and `backend/main.py`.
- [x] Run complete automated test suite across all 15 audit areas.
- [x] Update `README.md` with complete architecture, instructions, and Windows PowerShell commands.

---

## 5. Final GitHub Audit Corrections & Decoupling

Following the core refactor, final GitHub audit corrections were implemented to eliminate remaining inconsistencies:

1. **Complete Removal of Word Recognition from Active Pipeline**:
   - `ISLWordRecognizer` and `SentenceBuilder` are completely removed from the import and execution path of `backend/vision/live_pipeline.py`.
   - The active recognition modes are strictly `Alphabet (A–Z)` and `Numbers (0–9)`.
   - The UI radio selector exposes exclusively `["Alphabet (A–Z)", "Numbers (0–9)"]`.
   - All historical word-recognition files remain unimported and archived in `backend/recognition/`.
2. **Strict Signer-Independent Training Splits**:
   - Removed all random `train_test_split` fallbacks in `scripts/train_alphabet.py` and `scripts/train_numbers.py`.
   - Enforced a hard minimum of 3 unique signer IDs for training; scripts fail immediately with an informative error if signer diversity is insufficient.
   - Evaluated models record train signers, validation signers, test signers, and per-class test sample counts in `models/*/evaluation.json`.
3. **Number Label Normalization & Validation**:
   - Number model output is locked to exactly 10 classes (`0` to `9`).
   - Normalization of source labels like `TWO` to `2` occurs strictly during dataset validation/import via `isl_data_collection/storage.py`.
   - Out-of-range inputs (e.g. `10` or invalid strings) are rejected with `ValueError`.
4. **Model Asset Availability & Git Policy**:
   - Documented that `.keras` and `.task` files are excluded from git via `.gitignore` to keep the repository lightweight.
   - Provided `scripts/setup_assets.py` for automated MediaPipe task download and schema validation.
   - Live pipeline cleanly displays **"Model Not Available"** with instructions if assets are absent on disk.
5. **Test Suite Modernization**:
   - Unit tests use mocked models and schema validators.
   - Integration tests requiring gitignored model files use `@pytest.mark.skipif` to pass cleanly on fresh checkouts.
