# SignBridge AI — Model Training Status Report

**Date:** November 2026  
**Pipeline Standard:** MediaPipe Tasks 282-Dimensional Hand Landmarks $\rightarrow$ 30-Frame Resampling $\rightarrow$ Bidirectional LSTM Architecture  
**Execution Environment:** Python 3.11 / TensorFlow 2.16 (CPU AVX2+FMA)  
**Primary Focus:** Exclusive Real-Time Recognition of ISL English Alphabet (A–Z, 26 classes) & Numbers (0–9, 10 classes).

---

## 1. Architectural Standardization

Both active SignBridge AI models share a standardized feature extraction and temporal alignment pipeline:
- **Feature Vector:** 282 continuous floating-point values per time step:
  - Left Hand Slot: 63 absolute $(x,y,z)$ coordinates + 63 wrist-relative coordinates + 10 normalized finger joint angles + 4 fingertip distances = 140 features
  - Right Hand Slot: 63 absolute $(x,y,z)$ coordinates + 63 wrist-relative coordinates + 10 normalized finger joint angles + 4 fingertip distances = 140 features
  - Hand Presence Flags: 2 binary floats (`[left_present, right_present]`)
  - Total: $140 + 140 + 2 = 282$ floats
- **Temporal Window:** 30 consecutive frames ($T=30$), linearly resampled from variable-length video segments.
- **Input Tensor Shape:** `(batch_size, 30, 282)`
- **Core Architecture:** Bidirectional LSTM with spatial dropout, dense projection, and softmax classification.
- **Strict Gating:** Confidence threshold + top-1 vs top-2 margin gating to prevent flickering between close signs.

---

## 2. Active Model 1: ISL Number Recognition (`models/isl_number_model.keras`)

### 2.1 Training Configuration
- **Model Output:** 10 classes (`0` through `9`)
- **Dataset:** Official ISLRTC master signer videos (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`)
- **Total Master Samples:** 10 canonical sequences (1 per digit 0–9)
- **Signer Provenance:** Exactly 1 human master signer (`islrtc_master`). Artificial synthetic signers (`signer_sim_*`) have been completely removed.
- **Training Strategy:** Canonical samples with post-split spatial regularization (130 augmented training sequences, 10 validation, 10 test).
- **Hyperparameters:**
  - Optimizer: Adam ($lr = 0.001$)
  - Loss: Sparse Categorical Crossentropy
  - Batch Size: 16
  - Architecture: Input(30, 282) $\rightarrow$ BiLSTM(64) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(10, Softmax)
- **Evaluation Status:** Transparently reported as single-signer stratified split (`is_signer_independent: false`). Multi-signer benchmark requires recording $\ge 3$ signers via `isl_data_collection`.
- **Artifacts Saved:** `models/isl_number_model.keras`, `models/number/hand_sequence_lstm.keras`, `models/number_labels.json`, `models/number/evaluation.json`

---

## 3. Active Model 2: ISL Alphabet / Fingerspelling (`models/isl_fingerspelling_model.keras`)

### 3.1 Training Configuration
- **Model Output:** 26 classes (`A` through `Z`)
- **Dataset:** Hugging Face `kirandevraj/ISL-Fingerspelling` (CC BY-NC 4.0)
- **Total Instances:** 645 isolated letter sequences extracted from 59 continuous video segments
- **Signer-Independent Split:**
  - Training Signer: `signer_00` (194 samples)
  - Validation Signer: `signer_01` (126 samples)
  - Held-out Test Signer: `signer_02` (325 samples)
- **Signer-Independent Test Accuracy:** **58.46%** on unseen signer (`signer_02`) across all 26 alphabet classes (chance baseline = 3.85%).
- **Artifacts Saved:** `models/isl_fingerspelling_model.keras`, `models/alphabet/hand_sequence_lstm.keras`, `models/fingerspelling_labels.json`, `models/alphabet/evaluation.json`

---

## 4. Summary of Model Files

| Model Path | Task | Input Shape | Output Shape | Classes | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/isl_fingerspelling_model.keras` | ISL Alphabet | `(None, 30, 282)` | `(None, 26)` | A–Z (26) | Verified & Active |
| `models/isl_number_model.keras` | ISL Numbers | `(None, 30, 282)` | `(None, 10)` | 0–9 (10) | Verified & Active |
| `models/hand_landmarker.task` | Hand Tracking | Image / Video | 21 Landmarks | 2 Hands | Official MediaPipe |
