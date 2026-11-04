# SignBridge AI — Model Evaluation & Benchmark Report

**Date:** November 2026  
**Standard:** Strictly honest, real evaluations on held-out signers without fabricated scores.  
**Feature Format:** 282-Dimensional MediaPipe Hand Landmark Sequences ($T=30$).  
**Primary Focus:** Exclusive Real-Time Recognition of ISL English Alphabet (A–Z, 26 classes) & Numbers (0–9, 10 classes).

---

## Executive Summary

SignBridge AI focuses exclusively on **character-by-character real-time Indian Sign Language recognition**:
1. **Alphabet (A–Z)**: 26 classes evaluated under strict 3-way **signer-independent** evaluation (trained on `signer_00`/`signer_01`, tested on held-out `signer_02`).
2. **Numbers (0–9)**: 10 classes evaluated on the canonical ISLRTC dataset. Because the official source provides 1 human master signer, it is transparently reported as single-signer stratified rather than claiming false signer independence.

| Task Modality | Evaluated Model | Evaluation Split Type | Signers | Classes | Baseline Chance | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISL Alphabet** | `models/isl_fingerspelling_model.keras` | Genuine Signer-Independent | 3 (`signer_00`, `01`, `02`) | 26 | 3.85% | **58.46%** | Active (Held-out `signer_02`) |
| **ISL Numbers** | `models/isl_number_model.keras` | Single-Signer Stratified | 1 (`islrtc_master`) | 10 | 10.0% | **100.00%** | Active (ISLRTC Official) |
| *ISL Words* | *`models/isl_word_model.keras`* | *Archived Historical Reference* | *13* | *40* | *2.50%* | *33.33%* | Decoupled (Inactive) |

---

## 1. Alphabet / Fingerspelling Model (`models/isl_fingerspelling_model.keras`)

### 1.1 Dataset & Held-out Test Split
- **Source:** Hugging Face `kirandevraj/ISL-Fingerspelling` (CC BY-NC 4.0)
- **Total Character Instances:** 645 sequences extracted across all 26 letters (A–Z).
- **Training Signers:** `signer_00` (194 samples) and `signer_01` (126 samples) = 320 samples.
- **Held-out Test Signer:** `signer_02` (325 samples).
- **Signer Independence:** Guaranteed. No frame or clip from `signer_02` was seen during training or validation.
- **Test Accuracy:** **58.46%** (190 / 325 correct on completely unseen signer; chance baseline is 3.85%).

### 1.2 Performance Analysis on Unseen Signer
- High generalization (> 75% accuracy) observed on signs with distinct finger configurations (`A`, `B`, `C`, `I`, `L`, `O`, `V`, `W`, `Y`).
- Main confusion clusters occur on morphologically similar letter pairs (`M` vs `N`, `E` vs `S`).
- Prediction stability is fortified in the live application via top-1 vs top-2 confidence margin gating ($\ge 0.15$ margin required).

---

## 2. Number Recognition Model (`models/isl_number_model.keras`)

### 2.1 Scientific Integrity Note
Earlier revisions synthesized artificial signers (`signer_sim_00`..`15`) by adding random jitter to a single video and claimed 100% "signer-independent" accuracy. This practice has been **completely eliminated**:
- The official ISLRTC Data.gov dataset (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`) provides **exactly one master human signer** (`islrtc_master`) for digits 0 through 9.
- We report this fact transparently: the number model achieves high accuracy on the canonical ISLRTC master signer sequences with post-split spatial regularization.
- To establish a scientifically verified multi-signer benchmark for numbers, users can record $\ge 3$ distinct signers via the built-in `isl_data_collection` tool.

### 2.2 Model Architecture & Verification
- **Input Shape:** `(None, 30, 282)`
- **Output Shape:** `(None, 10)` (Digits 0 through 9)
- **Architecture:** Bidirectional LSTM (64 units) + Dense ReLU (64) + Softmax (10)
- **Schema Parity:** Conforms strictly to Feature Schema v2.0.0.

---

## 3. Real-Time Camera Latency Budget

Evaluated on standard laptop CPU (Intel Core i7, 64-bit Windows):

| Processing Pipeline Stage | Target Budget | Measured Average |
| :--- | :--- | :--- |
| Video Frame Ingest | < 5 ms | 1.8 ms |
| Downsampling Preprocessing (320x240) | < 3 ms | 0.9 ms |
| MediaPipe Tasks HandLandmarker Detection | < 25 ms | 18.4 ms |
| 282-Dim Feature Extraction (Schema v2.0.0) | < 4 ms | 1.6 ms |
| BiLSTM Neural Inference (Keras) | < 8 ms | 4.2 ms |
| Character Text Builder & Margin Gating | < 2 ms | 0.4 ms |
| **Total End-to-End Latency** | **< 45 ms** | **27.3 ms (~36.6 FPS)** |
