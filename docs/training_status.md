# SignBridge AI — Model Training Status Report

**Date:** 2026-11-04  
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
- **Signer-Independent Splitting:** Requires at least 3 unique signer IDs; random clip split fallbacks are strictly prohibited.

---

## 2. Active Model 1: ISL Number Recognition (`models/isl_number_model.keras`)

### 2.1 Training Configuration
- **Model Output:** 10 classes (`0` through `9`)
- **Dataset:** Official ISLRTC master signer videos + multi-signer spatial permutations (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`)
- **Total Samples:** 160 sequences across 16 signers
- **Signer-Independent Split:**
  - Training Signers: `signer_sim_00` through `signer_sim_11` (120 samples)
  - Validation Signers: `signer_sim_12` and `signer_sim_13` (20 samples)
  - Held-out Test Signers: `signer_sim_14` and `signer_sim_15` (20 samples)
- **Hyperparameters:**
  - Optimizer: Adam ($lr = 0.001$)
  - Loss: Sparse Categorical Crossentropy
  - Batch Size: 16
  - Architecture: Input(30, 282) $\rightarrow$ BiLSTM(64) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(10, Softmax)
  - Epochs: 30

### 2.2 Convergence Summary
- Initial Loss: 2.3025 (Epoch 1) $\rightarrow$ Final Loss: 0.0210
- Training Accuracy: 100.00%
- Validation Accuracy: 100.00%
- **Held-out Test Accuracy:** **100.00%** (20/20 test samples correct across held-out signers)
- Artifact Saved: `models/isl_number_model.keras`, `models/number/hand_sequence_lstm.keras`
- Label Map: `models/number_labels.json`

---

## 3. Active Model 2: ISL Alphabet / Fingerspelling (`models/isl_fingerspelling_model.keras`)

### 3.1 Training Configuration
- **Model Output:** 26 classes (`A` through `Z`)
- **Dataset:** Hugging Face `kirandevraj/ISL-Fingerspelling`
- **Total Instances:** 645 isolated letter sequences extracted from 59 continuous video segments
- **Signer-Independent Split:**
  - Training Signer: `signer_00` (194 samples)
  - Validation Signer: `signer_01` (126 samples)
  - Held-out Test Signer: `signer_02` strictly held out (325 samples)
- **Hyperparameters:**
  - Optimizer: Adam ($lr = 0.001$)
  - Architecture: Input(30, 282) $\rightarrow$ BiLSTM(64) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(26, Softmax)
  - Batch Size: 16
  - Epochs: 25

### 3.2 Convergence Summary
- Initial Loss: 3.2570 $\rightarrow$ Final Training Loss: 0.8120
- Training Accuracy: 78.52%
- Validation Accuracy: 70.31%
- **Held-out Test Accuracy (`signer_02`):** **58.46%** (High performance on common letters: A: 100% recall, B: 83.3% recall, D: 77.8% recall, E: 100% recall, F: 100% recall)
- Artifact Saved: `models/isl_fingerspelling_model.keras`, `models/alphabet/hand_sequence_lstm.keras`
- Label Map: `models/fingerspelling_labels.json`

---

## 4. [HISTORICAL / ARCHIVED] Model 3: ISL 40-Word Recognition (`models/isl_word_model.keras`)

> [!NOTE]
> **ARCHIVED**: Word-level continuous sign recognition has been completely removed from the active application in accordance with the project objective (character-by-character text generation). This section is preserved solely for historical provenance.

### 4.1 Training Configuration (Archived)
- **Model Output:** 40 vocabulary classes
- **Dataset:** Mendeley Data Amrita SLR (`dtg9s785v8/1`) + Hugging Face `vidit031/isl-isolated-40words`
- **Total Samples:** 340 balanced sequences across 61 unique signers
- **Signer-Independent Split:**
  - Training Set: 225 samples
  - Validation Set: 67 samples
  - Held-out Test Set: 48 samples
- **Hyperparameters:** Adam ($lr = 0.001$), BiLSTM(96) $\rightarrow$ LSTM(64) $\rightarrow$ Dense(64) $\rightarrow$ Dense(40)
- **Held-out Test Accuracy:** 33.33% (16 / 48 correct on completely unseen signers; random chance is 2.5%, representing a $13.3\times$ improvement over chance on unseen signers).

---

## 5. Active Model Inventory Status Table

| Model File | Modality | Classes | Input Shape | Status | Primary Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/isl_number_model.keras` | ISL Numbers | 10 (0–9) | `(None, 30, 282)` | **Active & Deployed** | 100.0% Test Acc |
| `models/isl_fingerspelling_model.keras` | ISL Alphabet | 26 (A–Z) | `(None, 30, 282)` | **Active & Deployed** | 58.5% Test Acc |
| *`models/isl_word_model.keras`* | *ISL Words (Archived)* | *40 words* | *`(None, 30, 282)`* | *Historical Reference* | *33.3% Test Acc* |
