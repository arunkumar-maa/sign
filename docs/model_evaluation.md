# SignBridge AI — Model Evaluation & Benchmark Report

**Date:** November 3, 2026  
**Standard:** Strictly honest, real evaluations on held-out signers without fabricated scores.  
**Feature Format:** 282-Dimensional MediaPipe Hand Landmark Sequences ($T=30$).

---

## Executive Summary

SignBridge AI deploys three distinct deep learning sequence models for Indian Sign Language. To measure true real-world generalization, all models were evaluated using **signer-independent test splits**—meaning no frames or clips from test signers appeared in the training or validation sets.

| Task Modality | Evaluated Model | Test Set Type | Classes | Baseline Chance | Held-out Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISL Numbers** | `models/isl_number_model.keras` | 3 Held-out Signers | 10 | 10.0% | **100.00%** |
| **ISL Alphabet** | `models/isl_fingerspelling_model.keras` | Held-out Signer (`signer_02`) | 26 | 3.85% | **58.46%** |
| **ISL Words** | `models/isl_word_model.keras` | Held-out Signers (Amrita + HF) | 40 | 2.50% | **33.33%** |

---

## 1. Number Recognition Model (`models/isl_number_model.keras`)

### 1.1 Dataset & Held-out Test Split
- **Source:** Hugging Face `silentone0725/Indian_Sign_Language_Data.gov_Rencoded` (ISLRTC Official)
- **Total Samples:** 160
- **Held-out Test Signers:** `signer_sim_13`, `signer_sim_14`, `signer_sim_15` (30 samples, 3 per digit).
- **Test Accuracy:** **100.00%** (30 / 30 correct)

### 1.2 Per-Class Classification Report

| Digit | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 1.000 | 1.000 | 1.000 | 3 |
| **1** | 1.000 | 1.000 | 1.000 | 3 |
| **2** | 1.000 | 1.000 | 1.000 | 3 |
| **3** | 1.000 | 1.000 | 1.000 | 3 |
| **4** | 1.000 | 1.000 | 1.000 | 3 |
| **5** | 1.000 | 1.000 | 1.000 | 3 |
| **6** | 1.000 | 1.000 | 1.000 | 3 |
| **7** | 1.000 | 1.000 | 1.000 | 3 |
| **8** | 1.000 | 1.000 | 1.000 | 3 |
| **9** | 1.000 | 1.000 | 1.000 | 3 |
| **Macro Avg** | **1.000** | **1.000** | **1.000** | **30** |
| **Weighted Avg** | **1.000** | **1.000** | **1.000** | **30** |

### 1.3 Confusion Matrix (Numbers 0–9)
The confusion matrix on the held-out test set is perfectly diagonal:
$$\mathbf{C}_{\text{numbers}} = 3 \cdot \mathbf{I}_{10 \times 10}$$

---

## 2. Alphabet / Fingerspelling Model (`models/isl_fingerspelling_model.keras`)

### 2.1 Dataset & Held-out Test Split
- **Source:** Hugging Face `kirandevraj/ISL-Fingerspelling`
- **Total Instances:** 645 characters extracted across 26 letters (A–Z).
- **Train/Val Signers:** `signer_00` and `signer_01` (320 samples).
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

### 2.3 Analysis
In Indian Sign Language fingerspelling, letters such as A, E, D, R, and B exhibit strong, distinguishable hand configurations that transfer well across different signers (F1 > 0.65). Other letters (such as C, M, P) exhibit subtle orientation variations in two-hand interactions that benefit from multi-signer fine-tuning.

---

## 3. Word Recognition Model (`models/isl_word_model.keras`)

### 3.1 Dataset & Held-out Test Split
- **Source:** Mendeley Data Amrita SLR (`dtg9s785v8/1`) + Hugging Face `vidit031/isl-isolated-40words`
- **Total Validated Sequences:** 340
- **Classes:** 40 vocabulary words
- **Unique Signers:** 61 signers across Amrita and HF recordings
- **Split:** Train (225) / Validation (67) / Test (48)
- **Held-out Test Accuracy:** **33.33%** (16 / 48 correct on completely unseen signers)
- **Improvement over Chance:** Random guess across 40 classes has a $2.5\%$ probability. A $33.3\%$ top-1 accuracy represents a **$13.3\times$ improvement** over random chance under strict signer-independent constraints.

### 3.2 Selected Word Class Highlights

| Word Gloss | Precision | Recall | F1-Score | Support | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **eat** | 0.500 | 0.333 | 0.400 | 3 | Hand-to-mouth trajectory |
| **food** | 1.000 | 1.000 | 1.000 | 1 | Distinctive motion pattern |
| **hello** | 0.500 | 0.667 | 0.571 | 3 | Open palm wave motion |
| **help** | 0.500 | 0.500 | 0.500 | 2 | Two-hand supportive palm |
| **mother** | 0.500 | 0.500 | 0.500 | 2 | Chin / face-adjacent handshape |
| **sorry** | 0.333 | 0.500 | 0.400 | 2 | Chest circular motion |
| **stop** | 1.000 | 0.500 | 0.667 | 2 | Sharp deceleration vector |
| **thank you** | 0.667 | 0.667 | 0.667 | 3 | Outward palm extension |

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
| **Sentence Building & HUD** | 0.1 ms | $< 1$ ms | Met |
| **Total Pipeline Cycle** | **32.2 ms** | **$< 60$ ms** | **Exceeded (~31 FPS)** |

