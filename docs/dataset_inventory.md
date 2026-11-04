# SignBridge AI — Dataset Inventory & Data Manifest Documentation

**Date:** November 2026  
**Standardized Feature Representation:** 282-dimensional MediaPipe hand landmarks (2 hands $\times$ [63 absolute + 63 wrist-relative + 10 joint angles + 4 fingertip distances] + 2 presence flags), resampled to 30 temporal frames.  
**Focus:** Strictly English Alphabet (A–Z, 26 classes) and Numbers (0–9, 10 classes).

---

## 1. Active Recognition Datasets

### 1.1 Dataset A — ISL Alphabet / Fingerspelling (26 Classes: A through Z)
- **Source Repository:** Hugging Face Datasets (`kirandevraj/ISL-Fingerspelling`)
- **Metadata Files:** `letter_annotations.csv`, `signer_split.csv`
- **Total Continuous Video Segments:** 1,308 segments (14,685 character annotations).
- **Curated Extracted Samples:** 645 isolated character sequences stored in `data/isl_signs/features/alphabet_*.npz`.
- **Classes Covered (26):** All letters `A` through `Z`.
- **Genuine Human Signers (3):**
  - `signer_00`: 194 samples (Train)
  - `signer_01`: 126 samples (Validation)
  - `signer_02`: 325 samples (Held-out Test)
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- **Storage Paths:**
  - Raw videos: `data/raw/isl_fingerspelling/videos/`
  - Extracted features: `data/isl_signs/features/alphabet_*.npz`
  - Manifest: `data/isl_signs/landmarks.jsonl`

### 1.2 Dataset B — ISL Numbers (10 Classes: 0 through 9)
- **Source Repository:** Hugging Face Datasets (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`)
- **Dataset Origin:** Official Indian Sign Language Research and Training Centre (ISLRTC) Data.gov.in dataset.
- **Selective Download:** Primary MP4 clips for digits `0` to `9` (no whole repository clone):
  - `0`: `Z/Zero.mp4`
  - `1`: `Numbers/1_One.mp4`
  - `2`: `Numbers/2_Two.mp4`
  - `3`: `Numbers/3_Three.mp4`
  - `4`: `Numbers/4_Four.mp4`
  - `5`: `Numbers/5_Five.mp4`
  - `6`: `Numbers/6_Six.mp4`
  - `7`: `Numbers/7_Seven.mp4`
  - `8`: `Numbers/8_Eight.mp4`
  - `9`: `Numbers/9_Nine.mp4`
- **Extracted Samples:** Exactly 10 genuine master sequences (shape: `30, 282`), stored in `data/isl_signs/features/number_islrtc_num_{0..9}.npz`.
- **Signer Provenance:** Exactly 1 human master signer (`islrtc_master`). All synthetic signers (`signer_sim_*`) have been permanently purged.
- **License:** Open Government Data (OGDL - India).
- **Storage Paths:**
  - Raw videos: `data/raw/islrtc_numbers/`
  - Extracted features: `data/isl_signs/features/number_islrtc_num_{0..9}.npz`
  - Manifest: `data/isl_signs/landmarks.jsonl`

---

## 2. Inactive / Archived Datasets (Historical Reference Only)

The following datasets were integrated during earlier development phases but are **not** part of the active Alphabet/Number recognition pipeline:
- **Amrita SLR Real-Time ISL Dataset** (Mendeley Data `dtg9s785v8/1`, CC BY 4.0, 50 word signs).
- **Hugging Face Isolated 40 Words** (`vidit031/isl-isolated-40words`, CC BY 4.0, 40 word signs).
- Retained strictly as offline reference files; unimported by the active live pipeline or server.

---

## 3. Standard Data Manifest Specification

All active training examples are catalogued in `data/isl_signs/landmarks.jsonl` conforming to the unified schema:

```json
{
  "sample_id": "islrtc_num_0",
  "category": "number",
  "label": "0",
  "signer_id": "islrtc_master",
  "session_id": "islrtc_official_dataset",
  "video_path": "data/raw/islrtc_numbers/islrtc_0_Zero.mp4",
  "features_path": "data/isl_signs/features/number_islrtc_num_0.npz",
  "source": "silentone0725/Indian_Sign_Language_Data.gov_Rencoded",
  "license": "OGDL - India"
}
```
