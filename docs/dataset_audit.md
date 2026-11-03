# SignBridge AI – Hugging Face Indian Sign Language Datasets Audit

**Date:** 2026-11-03  
**Audited Repositories:**
1. `vidit031/isl-isolated-40words`
2. `ISL500/ISL-DATA`
3. `ai4bharat/INCLUDE`
4. `pedroodb/glosl-isl`

---
## 1. Remote Dataset Catalog

| Dataset ID | License | Files Count | Downloads | Role / Purpose |
|---|---|---|---|---|
| `vidit031/isl-isolated-40words` | other | 1327 | 783 | Primary prototype training (40 classes, verified clips) |
| `ISL500/ISL-DATA` | unspecified | 100000 | 1209 | Large-scale research (500 classes, 159 GB archive) |
| `ai4bharat/INCLUDE` | cc-by-4.0 | 5 | 460 | Comprehensive benchmark (263 classes, 4292 videos) |
| `pedroodb/glosl-isl` | cc-by-nc-4.0 | 100000 | 2057 | Continuous sentence translation / glossing |

---
## 2. Provenance & License Review

### `vidit031/isl-isolated-40words`
- **Classes:** 40 core words (greetings, emergency, family, school, verbs).
- **Clips:** 642 MP4 video clips recorded by multiple native signers.
- **License / Provenance:** Curated from CISLR & ISL500 subsets for isolated word recognition.
- **Status in SignBridge AI:** Downloaded and verified locally (`data/raw/isl_isolated_40words/`).

### `ISL500/ISL-DATA`
- **Archive Size:** ~159 GB compressed.
- **Caution:** As per engineering guidelines, do not download the full 159 GB archive during prototype phase.

### `ai4bharat/INCLUDE`
- **Description:** High-quality multimodal dataset created by AI4Bharat across 263 classes.
- **Suitability:** Best for expanding vocabulary from 40 words to 260+ words in production.

### `pedroodb/glosl-isl`
- **Description:** Continuous sign sentences aligned with spoken English/Hindi glosses.
- **Suitability:** Target for future continuous sentence translation models.