# SignBridge AI — Model Evaluation & Benchmark Report

**Date:** 2026-11-04  
**Standard:** Strictly honest, real evaluations on held-out signers without fabricated scores.  
**Feature Format:** 282-Dimensional MediaPipe Hand Landmark Sequences ($T=30$).  
**Primary Focus:** Exclusive Real-Time Recognition of ISL English Alphabet (A–Z, 26 classes) & Numbers (0–9, 10 classes).

---

## Executive Summary

SignBridge AI focuses exclusively on **character-by-character real-time Indian Sign Language recognition**. To measure true real-world generalization, both active models were evaluated using **genuine signer-independent test splits**—meaning no frames or clips from test signers appeared in the training or validation sets.

| Task Modality | Evaluated Model | Test Set Type | Classes | Baseline Chance | Held-out Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISL Numbers (Active)** | `models/isl_number_model.keras` | Held-out Signers (`signer_sim_14`, `15`) | 10 | 10.0% | **100.00%** |
| **ISL Alphabet (Active)** | `models/isl_fingerspelling_model.keras` | Held-out Signer (`signer_02`) | 26 | 3.85% | **58.46%** |
| *ISL Words (Archived)* | *`models/isl_word_model.keras`* | *Held-out Signers (Amrita + HF)* | *40* | *2.50%* | *33.33%* |

---

## 1. Number Recognition Model (`models/isl_number_model.keras`)

### 1.1 Dataset & Held-out Test Split
- **Source:** Hugging Face `silentone0725/Indian_Sign_Language_Data.gov_Rencoded` (ISLRTC Official)
- **Total Samples:** 160 across 16 signers
- **Held-out Test Signers:** `signer_sim_14`, `signer_sim_15` (20 samples, 2 per digit).
- **Test Accuracy:** **100.00%** (20 / 20 correct)

### 1.2 Per-Class Classification Report

| Digit | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 1.000 | 1.000 | 1.000 | 2 |
| **1** | 1.000 | 1.000 | 1.000 | 2 |
| **2** | 1.000 | 1.000 | 1.000 | 2 |
| **3** | 1.000 | 1.000 | 1.000 | 2 |
| **4** | 1.000 | 1.000 | 1.000 | 2 |
| **5** | 1.000 | 1.000 | 1.000 | 2 |
| **6** | 1.000 | 1.000 | 1.000 | 2 |
| **7** | 1.000 | 1.000 | 1.000 | 2 |
| **8** | 1.000 | 1.000 | 1.000 | 2 |
| **9** | 1.000 | 1.000 | 1.000 | 2 |
| **Macro Avg** | **1.000** | **1.000** | **1.000** | **20** |
| **Weighted Avg** | **1.000** | **1.000** | **1.000** | **20** |

---

## 2. Alphabet / Fingerspelling Model (`models/isl_fingerspelling_model.keras`)

### 2.1 Dataset & Held-out Test Split
- **Source:** Hugging Face `kirandevraj/ISL-Fingerspelling`
- **Total Instances:** 645 characters extracted across 26 letters (A–Z).
- **Train/Val Signers:** `signer_00` (194 samples) and `signer_01` (126 samples).
- **Held-out Test Signer:** `signer_02` (325 samples).
- **Test Accuracy:** **58.46%** (190 / 325 correct on unseen signer).

### 2.2 Selected High-Performing Classes (Held-out Signer 02)

| Letter | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **A** | 0.744 | 1.000 | 0.853 | 61 |
| **B** | 0.556 | 0.833 | 0.667 | 6 |
| **D** | 0.583 | 0.778 | 0.667 | 18 |
| **E** | 0.586 | 1.000 | 0.739 | 17 |
| **F** | 1.000 | 0.444 | 0.615 | 9 |
| **H** | 0.462 | 0.667 | 0.545 | 9 |
| **K** | 1.000 | 0.600 | 0.750 | 5 |
| **L** | 0.600 | 0.750 | 0.667 | 8 |
| **N** | 0.667 | 0.571 | 0.615 | 7 |
| **O** | 0.625 | 0.625 | 0.625 | 8 |
| **Q** | 0.750 | 0.600 | 0.667 | 5 |
| **R** | 0.833 | 0.833 | 0.833 | 6 |
| **T** | 0.800 | 0.571 | 0.667 | 7 |
| **U** | 0.500 | 0.750 | 0.600 | 8 |
| **V** | 0.500 | 0.500 | 0.500 | 6 |
| **W** | 0.571 | 0.571 | 0.571 | 7 |

---

## 3. [HISTORICAL / ARCHIVED] Word Recognition Model (`models/isl_word_model.keras`)

> [!NOTE]
> **ARCHIVED**: Word-level continuous sign recognition has been completely removed from the active application in accordance with the project objective (character-by-character text generation). This benchmark is preserved solely for historical documentation.

### 3.1 Dataset & Held-out Test Split (Archived)
- **Source:** Mendeley Data Amrita SLR (`dtg9s785v8/1`) + Hugging Face `vidit031/isl-isolated-40words`
- **Total Validated Sequences:** 340 across 61 signers
- **Classes:** 40 vocabulary words
- **Held-out Test Accuracy:** **33.33%** (16 / 48 correct on completely unseen signers; $13.3\times$ improvement over random chance).

---

## 4. Operational Inference Latencies

Evaluated on Intel CPU using MediaPipe Tasks (320x240 detector resolution) and Keras BiLSTM inference:

| Pipeline Stage | Mean Latency (ms) | Target SLA | Status |
| :--- | :--- | :--- | :--- |
| **Camera Ingest** | 1.8 ms | $< 5$ ms | Met |
| **Downsample & Preprocess** | 0.9 ms | $< 3$ ms | Met |
| **MediaPipe Hand Landmarker** | 24.2 ms | $< 35$ ms | Met |
| **282-Dim Feature Extraction** | 0.4 ms | $< 2$ ms | Met |
| **Keras Sequence Inference** | 4.8 ms | $< 15$ ms | Met |
| **Character Building & HUD** | 0.1 ms | $< 1$ ms | Met |
| **Total Pipeline Cycle** | **32.2 ms** | **$< 60$ ms** | **Exceeded (~31 FPS)** |
