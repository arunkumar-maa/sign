# SignBridge AI — Dataset License Audit & Attribution Compliance

**Date:** November 3, 2026  
**Module:** `isl_data_collection` / `data/`  
**Compliance Standard:** Open-Source, Non-Commercial Educational, Academic Research & Hackathon Deployment

---

## 1. Overview of Licensure Across Staged Datasets

To ensure legal clarity, intellectual property compliance, and ethical data governance, every pre-training dataset ingested into SignBridge AI has been inventoried with its governing license terms, provenance, attribution requirements, and deployment constraints.

| Dataset Identifier | Governing License | Commercial Permissibility | Derivative Work & Model Distribution | Mandatory Attribution Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Amrita SLR Real-Time ISL Dataset** (`dtg9s785v8/1`) | **CC BY 4.0** | Permitted with attribution | Permitted; weights can be distributed | Required (Amrita WNA) |
| **HF `vidit031/isl-isolated-40words`** | **CC BY 4.0** / Open Access | Permitted with attribution | Permitted; weights can be distributed | Required (Vidit / Contributors) |
| **HF `kirandevraj/ISL-Fingerspelling`** | **CC BY-NC 4.0** | Non-Commercial Only | Permitted for non-commercial research/hackathons | Required (Kiran Devraj et al.) |
| **ISLRTC Government Video Data** (`silentone0725/...`) | **OGDL India** | Permitted with attribution | Permitted under National Data Sharing & Accessibility Policy | Required (ISLRTC, Govt. of India) |

---

## 2. Detailed License Audits

### 2.1 Mendeley Data: Amrita SLR Real-Time ISL Dataset
- **Identifier:** `dtg9s785v8/1`
- **Authors:** Amrita Center for Wireless Networks and Applications (Amrita WNA), Amrita Vishwa Vidyapeetham, India.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Permissions:**
  - **Share:** Copy and redistribute the material in any medium or format.
  - **Adapt:** Remix, transform, and build upon the material for any purpose, even commercially.
- **Obligations:**
  - **Attribution:** Appropriate credit must be provided with a link to the license and indications if changes were made.
  - **No Additional Restrictions:** No legal terms or technological measures may be applied that restrict others from doing anything the license permits.
- **SignBridge AI Compliance:**
  - Sourced directly from Mendeley Data S3 repository.
  - Metadata manifest records dataset source as `amrita_slr_dtg9s785v8_1`.
  - Feature extraction script records origin per sample in `data/isl_signs/landmarks.jsonl`.

### 2.2 Hugging Face: `vidit031/isl-isolated-40words`
- **Identifier:** `vidit031/isl-isolated-40words`
- **Curator:** Vidit / Open ISL Community.
- **License:** CC BY 4.0.
- **Permissions & Obligations:**
  - Fully compatible with CC BY 4.0 terms.
  - Retains per-signer metadata in `metadata.csv`.
- **SignBridge AI Compliance:**
  - Filtered to eliminate anomalous or "Needs Manual Review" samples.
  - Retains sample IDs and signer provenance (`hf_signer_01` .. `hf_signer_13`).
  - Stored in `data/raw/isl_isolated_40words/`.

### 2.3 Hugging Face: `kirandevraj/ISL-Fingerspelling`
- **Identifier:** `kirandevraj/ISL-Fingerspelling`
- **Authors:** Kiran Devraj et al.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- **Permissions:**
  - **Share & Adapt:** Permitted for academic, research, hackathon, and educational demonstrations.
- **Restrictions:**
  - **NonCommercial:** You may not use the material for commercial purposes without explicit dual-licensing from the copyright holder.
- **SignBridge AI Compliance:**
  - SignBridge AI is an open-source, accessible assistive technology prototype created for educational and hackathon demonstration purposes.
  - The trained fingerspelling model (`models/isl_fingerspelling_model.keras`) inherits the NonCommercial covenant for commercial distribution.

### 2.4 ISLRTC: Indian Sign Language Data.gov Rencoded
- **Identifier:** `silentone0725/Indian_Sign_Language_Data.gov_Rencoded`
- **Originating Authority:** Indian Sign Language Research and Training Centre (ISLRTC), Department of Empowerment of Persons with Disabilities, Ministry of Social Justice and Empowerment, Government of India.
- **License:** Open Government Data License (OGDL) India.
- **Permissions:**
  - Use, adapt, download, distribute, and transmit data for research, commercial, or non-commercial purposes.
- **Obligations:**
  - Acknowledge the data provider, source, and license by explicitly mentioning ISLRTC and data.gov.in.
  - Data must not be presented in a misleading or derogatory manner.
- **SignBridge AI Compliance:**
  - Attribution included in documentation, UI manifests, and training reports.
  - Video samples stored in `data/raw/islrtc_numbers/`.

---

## 3. Dataset Mixing & Dual-License Harmonization

| Combination | Components | Composite Artifact | Resulting License |
| :--- | :--- | :--- | :--- |
| **ISL Words** | Amrita SLR (CC BY 4.0) + HF 40 words (CC BY 4.0) | `models/isl_word_model.keras` | **CC BY 4.0** |
| **ISL Alphabet** | HF ISL-Fingerspelling (CC BY-NC 4.0) | `models/isl_fingerspelling_model.keras` | **CC BY-NC 4.0** |
| **ISL Numbers** | ISLRTC Data.gov (OGDL India) | `models/isl_number_model.keras` | **OGDL India / Open Access** |

> [!NOTE]
> Because the models for Words, Alphabet, and Numbers are trained into separate `.keras` weight files, there is **zero license contagion**. Commercial applications may freely bundle the Word and Number models while requiring a commercial alternative for the Fingerspelling model if monetized.

---

## 4. Formal Attribution Notice

SignBridge AI gratefully acknowledges the following research groups and organizations for providing open datasets:
1. **Amrita Center for Wireless Networks and Applications (Amrita WNA)** for the Amrita SLR Real-Time ISL Dataset (*Mendeley Data*, doi:10.17632/dtg9s785v8.1).
2. **Vidit and Contributors** for the `isl-isolated-40words` dataset on Hugging Face.
3. **Kiran Devraj and Collaborators** for the `ISL-Fingerspelling` dataset on Hugging Face.
4. **Indian Sign Language Research and Training Centre (ISLRTC)**, Government of India, for the official Indian Sign Language dictionary resources.

