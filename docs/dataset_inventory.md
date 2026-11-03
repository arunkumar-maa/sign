# SignBridge AI — Pre-Training Dataset Inventory

**Date:** November 3, 2026  
**Module:** `isl_data_collection` / `data/`  
**Standardized Feature Representation:** 282-dimensional MediaPipe hand landmarks (2 hands $\times$ [63 absolute + 63 wrist-relative + 10 joint angles + 4 fingertip distances] + 2 presence flags), resampled to 30 temporal frames.

---

## Executive Summary

SignBridge AI utilizes verified Indian Sign Language (ISL) datasets spanning three dedicated functional modalities:
1. **Word Recognition (40 ISL Vocabulary Classes)**: Blended multi-source dataset combining Mendeley Data Amrita SLR and Hugging Face `vidit031/isl-isolated-40words`.
2. **Alphabet / Fingerspelling Recognition (26 Classes, A–Z)**: Continuous and isolated segments from Hugging Face `kirandevraj/ISL-Fingerspelling`.
3. **Number Recognition (10 Classes, 0–9)**: Standard Indian Sign Language Research and Training Centre (ISLRTC) videos from Hugging Face `silentone0725/Indian_Sign_Language_Data.gov_Rencoded`.

---

## 1. Word Recognition Datasets (Vocabulary: 40 Signs)

### 1.1 Dataset 1A — Amrita SLR Real-Time ISL Dataset (Primary)
- **Source Repository:** Mendeley Data (`dtg9s785v8/1`)
- **Direct S3 URL:** `https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/21487202-e9d2-4f71-940c-f882f1f72695`
- **Total Archive Size:** 1.36 GB (1,465,051,081 bytes)
- **Original Content:** 50 ISL sign classes, 7 deaf signers, 1080p Full HD video @ 28 FPS.
- **Integrated Subset:** 17 classes matching SignBridge 40-word vocabulary:
  - `brother`, `drink`, `eat`, `father`, `friend`, `hello`, `help`, `mother`, `sit`, `sorry`, `stand`, `stop`, `student`, `teacher`, `thank you`, `today`, `write`.
- **Selected Staged Samples:** 146 video clips (8–21 per class).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Local Storage Path:** `data/raw/amrita_slr/`

### 1.2 Dataset 1B — Hugging Face Isolated 40 Words (Secondary)
- **Source Repository:** Hugging Face Datasets (`vidit031/isl-isolated-40words`)
- **Total Clips:** 642 MP4 video clips across 40 classes.
- **Accepted / Validated Clips:** 578 high-confidence clips filtered against `metadata.csv` (excluding "Needs Manual Review" or anomalous clips).
- **Signers:** 13 distinct signers (`signer_01` through `signer_13`).
- **Resolution & Frame Rate:** 720p / 1080p, 30 FPS, H.264 video encoding.
- **Classes Covered (40):**
  `brother`, `come`, `drink`, `eat`, `father`, `food`, `friend`, `go`, `goodbye`, `he`, `hello`, `help`, `home`, `hospital`, `market`, `me`, `mother`, `no`, `okay`, `please`, `read`, `school`, `she`, `sister`, `sit`, `sorry`, `stand`, `stop`, `student`, `tea`, `teacher`, `thank you`, `today`, `water`, `what`, `when`, `where`, `write`, `yes`, `you`.
- **License:** CC BY 4.0 / Open Access.
- **Local Storage Path:** `data/raw/isl_isolated_40words/`

### 1.3 Word Dataset Fusion & Balancing Strategy
- Raw merge produces 724 candidate clips.
- Amrita SLR specifically bolsters classes under-represented in the HF dataset (`brother`, `sorry`, `stand`, `stop`, `write`).
- Balanced sampling caps per-class counts at 10 clips maximum, yielding 343 high-quality clips across all 40 vocabulary classes.
- Signer-independent splitting reserves held-out signers for validation and test splits.

---

## 2. Alphabet / Fingerspelling Dataset (26 Classes, A–Z)

- **Source Repository:** Hugging Face Datasets (`kirandevraj/ISL-Fingerspelling`)
- **Metadata Files:** `letter_annotations.csv`, `signer_split.csv`
- **Total Repository Size:** ~1,308 continuous video segments (14,685 character instances across 3 signers).
- **Curated Download:** 59 targeted continuous segments downloaded into `data/raw/isl_fingerspelling/videos/`.
- **Extracted Instances:** 645 isolated character sequences extracted with exact frame start/end timestamps from `letter_annotations.csv`.
- **Alphabet Coverage:** All 26 letters (A through Z).
- **Signers & Split:**
  - `signer_00`: 168 instances (Training / Validation)
  - `signer_01`: 152 instances (Training / Validation)
  - `signer_02`: 325 instances (Held-out Test Set)
- **Resolution & Frame Rate:** 1920x1080 (1080p), 30 FPS.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- **Local Storage Path:** `data/raw/isl_fingerspelling/`

---

## 3. Number Recognition Dataset (10 Classes, Digits 0–9)

- **Source Repository:** Hugging Face Datasets (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`)
- **Authority:** Indian Sign Language Research and Training Centre (ISLRTC), Department of Empowerment of Persons with Disabilities, Ministry of Social Justice and Empowerment, Government of India.
- **Download Strategy:** Selective download of individual number video files (`Z/Zero.mp4`, `Numbers/1_One.mp4` through `Numbers/9_Nine.mp4`, and 4 regional variations). No bulk 75 GB download.
- **Classes Covered (10):** Digits `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`.
- **Sample Generation:** 160 total samples (16 samples per digit), combining official ISLRTC sign video recordings and multi-signer spatial permutations.
- **Signers & Split:**
  - Train Set: 104 samples (Signers: official + `signer_sim_01` to `signer_sim_10`)
  - Validation Set: 26 samples (Signers: `signer_sim_11` and `signer_sim_12`)
  - Held-out Test Set: 30 samples (Signers: `signer_sim_13`, `signer_sim_14`, `signer_sim_15`)
- **Resolution & Frame Rate:** 1280x720, 30 FPS.
- **License:** Open Government Data License (OGDL) India.
- **Local Storage Path:** `data/raw/islrtc_numbers/`

---

## 4. Dataset Summary Matrix

| Category | Primary Dataset | Secondary Dataset | Classes | Extracted Samples | Signers | Split Strategy | Target Model |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISL Words** | Mendeley Data `dtg9s785v8/1` | HF `isl-isolated-40words` | 40 words | 343 clips | 20 (7 Amrita + 13 HF) | Signer-Independent Group Split | `models/isl_word_model.keras` |
| **ISL Alphabet** | HF `ISL-Fingerspelling` | N/A | 26 letters (A–Z) | 645 segments | 3 (`signer_00`, `01`, `02`) | Held-out Signer (`signer_02`) | `models/isl_fingerspelling_model.keras` |
| **ISL Numbers** | HF `Indian_Sign_Language_Data.gov_Rencoded` (ISLRTC) | Regional videos | 10 digits (0–9) | 160 sequences | 16 (1 master + 15 simulated) | Held-out Signers (`sim_13..15`) | `models/isl_number_model.keras` |

---

## 5. Storage Directory Structure

```
data/
├── isl_signs/
│   ├── features/                  # Cached 282-dim .npz files (resampled 30 frames)
│   ├── videos/                    # Staged raw video recordings
│   ├── landmarks.jsonl            # Landmark extraction manifest
│   └── samples.jsonl              # Sample metadata manifest
└── raw/
    ├── amrita_slr/                # Mendeley Data Amrita SLR clips
    ├── isl_fingerspelling/        # HF kirandevraj/ISL-Fingerspelling
    │   ├── letter_annotations.csv
    │   ├── signer_split.csv
    │   └── videos/
    ├── isl_isolated_40words/      # HF vidit031/isl-isolated-40words
    │   ├── metadata.csv
    │   └── videos/
    └── islrtc_numbers/            # HF silentone0725 ISLRTC number videos
```

