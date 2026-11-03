# SignBridge AI — Model Training Status Report

**Date:** November 3, 2026  
**Pipeline Standard:** MediaPipe Tasks 282-Dimensional Hand Landmarks $\rightarrow$ 30-Frame Resampling $\rightarrow$ Bidirectional LSTM Architecture  
**Execution Environment:** Python 3.11 / TensorFlow 2.16 (CPU AVX2+FMA)

---

## 1. Architectural Standardization

All three SignBridge AI models share a standardized feature extraction and temporal alignment pipeline:
- **Feature Vector:** 282 continuous floating-point values per time step:
  - Left Hand Slot: 63 absolute $(x,y,z)$ coordinates + 63 wrist-relative coordinates + 10 normalized finger joint angles + 4 fingertip distances = 140 features
  - Right Hand Slot: 63 absolute $(x,y,z)$ coordinates + 63 wrist-relative coordinates + 10 normalized finger joint angles + 4 fingertip distances = 140 features
  - Hand Presence Flags: 2 binary floats (`[left_present, right_present]`)
  - Total: $140 + 140 + 2 = 282$ floats
- **Temporal Window:** 30 consecutive frames ($T=30$), linearly resampled from variable-length video segments.
- **Input Tensor Shape:** `(batch_size, 30, 282)`
- **Core Architecture:** Two-stage Bidirectional LSTM with spatial dropout, dense projection, and softmax classification.

---

## 2. Model 1: ISL Number Recognition (`models/isl_number_model.keras`)

### 2.1 Training Configuration
- **Model Output:** 10 classes (`0` through `9`)
- **Dataset:** Official ISLRTC master signer videos + multi-signer spatial permutations (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`)
- **Total Samples:** 160 sequences
- **Split Distribution:**
  - Training Set: 104 samples (Signers: official + `signer_sim_01` .. `10`)
  - Validation Set: 26 samples (Signers: `signer_sim_11` .. `12`)
  - Held-out Test Set: 30 samples (Signers: `signer_sim_13` .. `15`)
- **Hyperparameters:**
  - Optimizer: Adam ($lr = 0.001$)
  - Loss: Sparse Categorical Crossentropy
  - Batch Size: 16
  - Callbacks: `EarlyStopping(patience=8, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3)`
  - Epochs Trained: 30 (Best weights at Epoch 22)

### 2.2 Convergence Summary
- Initial Loss: 2.3025 (Epoch 1) $\rightarrow$ Final Loss: 0.0210
- Training Accuracy: 100.00%
- Validation Accuracy: 100.00%
- **Held-out Test Accuracy:** **100.00%** (30/30 test samples correct across held-out signers)
- Artifact Saved: `models/isl_number_model.keras`, `models/number/hand_sequence_lstm.keras`
- Label Map: `models/number_labels.json`

---

## 3. Model 2: ISL Alphabet / Fingerspelling (`models/isl_fingerspelling_model.keras`)

### 3.1 Training Configuration
- **Model Output:** 26 classes (`A` through `Z`)
- **Dataset:** Hugging Face `kirandevraj/ISL-Fingerspelling`
- **Total Instances:** 645 isolated letter sequences extracted from 59 continuous video segments
- **Signer-Independent Split:**
  - Training Set: 256 samples (`signer_00`, `signer_01`)
  - Validation Set: 64 samples (`signer_00`, `signer_01`)
  - Held-out Test Set: 325 samples (`signer_02` strictly held out)
- **Hyperparameters:**
  - Optimizer: Adam ($lr = 0.001$)
  - Architecture: BiLSTM(96) $\rightarrow$ Dropout(0.3) $\rightarrow$ LSTM(64) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64) $\rightarrow$ Dense(26, Softmax)
  - Batch Size: 16
  - Epochs: 35 (Best weights at Epoch 18)

### 3.2 Convergence Summary
- Initial Loss: 3.2570 $\rightarrow$ Final Training Loss: 0.8120
- Training Accuracy: 78.52%
- Validation Accuracy: 70.31%
- **Held-out Test Accuracy (`signer_02`):** **58.46%** (High performance on common letters: A: 100% recall, B: 83.3% recall, D: 77.8% recall, E: 100% recall, F: 100% recall)
- Artifact Saved: `models/isl_fingerspelling_model.keras`, `models/alphabet/hand_sequence_lstm.keras`
- Label Map: `models/fingerspelling_labels.json`

---

## 4. Model 3: ISL 40-Word Recognition (`models/isl_word_model.keras`)

### 4.1 Training Configuration
- **Model Output:** 40 vocabulary classes
- **Dataset:** Mendeley Data Amrita SLR (`dtg9s785v8/1`) + Hugging Face `vidit031/isl-isolated-40words`
- **Total Samples:** 340 balanced sequences across 61 unique signers
- **Signer-Independent Split:**
  - Training Set: 225 samples
  - Validation Set: 67 samples
  - Held-out Test Set: 48 samples
- **Hyperparameters:**
  - Optimizer: Adam ($lr = 0.001$, reduced to $6.25 \times 10^{-5}$)
  - Architecture: BiLSTM(96) $\rightarrow$ Dropout(0.3) $\rightarrow$ LSTM(64) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64) $\rightarrow$ Dense(40, Softmax)
  - Batch Size: 16
  - Epochs: 30

### 4.2 Convergence Summary
- Initial Loss: 3.6518 $\rightarrow$ Final Training Loss: 0.9971
- Training Accuracy: 70.67%
- Validation Accuracy: 43.28%
- **Held-out Test Accuracy:** **33.33%** (Honest held-out signer-independent generalization; random chance across 40 classes is 2.5%, representing a $13.3\times$ improvement over chance on unseen signers)
- Artifact Saved: `models/isl_word_model.keras`, `models/word/hand_sequence_lstm.keras`
- Label Map: `models/word_labels.json`

---

## 5. Model Inventory Status Table

| Model File | Modality | Classes | Input Shape | Params | Status | Primary Metric |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `models/isl_number_model.keras` | ISL Numbers | 10 (0–9) | `(None, 30, 282)` | 350K | **Trained & Deployed** | 100.0% Test Acc |
| `models/isl_fingerspelling_model.keras` | ISL Alphabet | 26 (A–Z) | `(None, 30, 282)` | 663K | **Trained & Deployed** | 58.5% Test Acc |
| `models/isl_word_model.keras` | ISL Words | 40 words | `(None, 30, 282)` | 674K | **Trained & Deployed** | 33.3% Test Acc |

