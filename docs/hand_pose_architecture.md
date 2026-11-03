# SignBridge AI — Hand Pose Based ISL Recognition Architecture

## 1. Architectural Overview & Paradigm Shift

In accordance with user requirements, the recognition architecture of SignBridge AI has been fundamentally transitioned from multi-modal tracking (hands + face + body pose) to a **Hand Pose Based Recognition System**.

### Core Changes:
- **Zero Full-Body Pose Dependency:** Completely removed shoulder tracking, elbow tracking, and torso tracking from the recognition pipeline.
- **Zero Facial Mesh Dependency:** Completely removed face landmark extraction from the feature pipeline.
- **Strictly Hand & Finger Focused:** The recognition models and feature extractors operate exclusively on 21 3D hand landmarks per hand, capturing fine-grained finger articulation, joint kinematics, inter-digit Euclidean distances, palm orientation, and inter-hand spatial interactions.

---

## 2. Hand Pose Feature Vector Specifications

The feature extraction pipeline is implemented in [`backend/vision/hand_pose_features.py`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/backend/vision/hand_pose_features.py).

### A. Per-Hand Features (104 Dimensions):
1. **Normalized 3D Coordinates (63 dims):**
   - Centered at the wrist (Landmark 0): $\mathbf{L}'_i = \mathbf{L}_i - \mathbf{L}_0$.
   - Scale-normalized by palm dimension $d_{\text{scale}} = \|\mathbf{L}_9 - \mathbf{L}_0\|_2$ (wrist to middle MCP).
2. **Finger Joint Angles (15 dims):**
   - 3 angles per finger computed via 3D vector dot products $\theta = \arccos\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}\right) \in [0, \pi]$:
     - Thumb: Wrist $\to$ CMC $\to$ MCP, CMC $\to$ MCP $\to$ IP, MCP $\to$ IP $\to$ TIP.
     - Index: Wrist $\to$ MCP $\to$ PIP, MCP $\to$ PIP $\to$ DIP, PIP $\to$ DIP $\to$ TIP.
     - Middle: Wrist $\to$ MCP $\to$ PIP, MCP $\to$ PIP $\to$ DIP, PIP $\to$ DIP $\to$ TIP.
     - Ring: Wrist $\to$ MCP $\to$ PIP, MCP $\to$ PIP $\to$ DIP, PIP $\to$ DIP $\to$ TIP.
     - Pinky: Wrist $\to$ MCP $\to$ PIP, MCP $\to$ PIP $\to$ DIP, PIP $\to$ DIP $\to$ TIP.
3. **Finger Distance Metrics (12 dims):**
   - Scale-normalized Euclidean distances:
     - Thumb tip to all 4 other fingertips (Index, Middle, Ring, Pinky).
     - Adjacent fingertip distances (Index-Middle, Middle-Ring, Ring-Pinky).
     - Fingertip to wrist distances for all 5 digits.
4. **Handshape Extension States (5 dims):**
   - Continuous state per digit: $1.0$ (Extended), $0.5$ (Bent), $0.0$ (Folded into palm).
   - Evaluated using digit tip-to-wrist vs. PIP-to-wrist ratios and joint straightness.
5. **Palm Orientation & Spatial Position (9 dims):**
   - Palm Normal unit vector $\mathbf{n} = \frac{\mathbf{u} \times \mathbf{v}}{\|\mathbf{u} \times \mathbf{v}\|}$ (3 dims).
   - Palm Pointing unit vector $\mathbf{d} = \frac{\mathbf{L}_9 - \mathbf{L}_0}{\|\mathbf{L}_9 - \mathbf{L}_0\|}$ (3 dims).
   - Unnormalized wrist coordinate $(x_0, y_0, z_0)$ for spatial tracking (3 dims).

### B. Unified Two-Hand Representation (216 Dimensions):
$$\mathbf{X} = [\mathbf{F}_{\text{left}} (104), \; \mathbf{F}_{\text{right}} (104), \; \mathbf{P}_{\text{presence}} (2), \; \mathbf{R}_{\text{relative}} (6)]$$
- **Left Hand Features:** 104 dims (zeros if missing).
- **Right Hand Features:** 104 dims (zeros if missing).
- **Presence Flags:** $[p_{\text{left}}, p_{\text{right}}] \in \{0.0, 1.0\}^2$ (2 dims).
- **Relative Two-Hand Spatial Geometry (6 dims):**
  - Inter-wrist displacement vector: $\mathbf{r}_{\text{wrist}} = \mathbf{W}_{\text{right}} - \mathbf{W}_{\text{left}}$ (3 dims).
  - Inter-wrist Euclidean distance: $\|\mathbf{r}_{\text{wrist}}\|_2$ (1 dim).
  - Minimum distance between any left fingertip and right fingertip (1 dim).
  - Palm normal alignment dot product: $\mathbf{n}_{\text{left}} \cdot \mathbf{n}_{\text{right}}$ (1 dim).

---

## 3. Dataset & Model Training Pipeline

- **Source Dataset:** `data/raw/isl_isolated_40words/` (642 MP4 video clips across 40 ISL vocabulary classes).
- **Stratified Partitioning:** Stratified per-class split ensuring 100% vocabulary coverage across splits:
  - Train: 430 videos + jitter/scale augmentation $\to$ 1,290 sequences.
  - Validation: 108 videos $\to$ 108 sequences.
  - Test: 108 videos $\to$ 108 sequences.

### A. Dynamic Hand Movement Recognizer
- **Script:** [`ml/train_dynamic_model.py`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/ml/train_dynamic_model.py)
- **Model Path:** [`models/hand_pose/isl_dynamic_handpose.keras`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/models/hand_pose/isl_dynamic_handpose.keras)
- **Architecture:** Bidirectional LSTM ($128 \to 64$) with LayerNormalization, Dropout, and Dense classification head.
- **Input:** $(30, 216)$ sliding sequence.
- **Evaluation Accuracy:** **53.70%** on held-out test split across 40 classes (Random baseline: 2.50%).
- **Inference Latency:** **31.29 ms/sequence**.

### B. Static Hand Pose Classifier
- **Script:** [`ml/train_static_model.py`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/ml/train_static_model.py)
- **Model Path:** [`models/hand_pose/isl_static_handpose.keras`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/models/hand_pose/isl_static_handpose.keras)
- **Architecture:** Multi-Layer Perceptron ($256 \to 128 \to 64$) with BatchNormalization and Dropout.
- **Input:** $(216,)$ single keyframe feature vector.
- **Evaluation Accuracy:** **46.30%** on held-out test split.
- **Inference Latency:** **0.80 ms/sample**.

### C. Evaluation Artifacts
- **Labels Dictionary:** [`models/hand_pose/labels.json`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/models/hand_pose/labels.json)
- **Comprehensive Evaluation Report:** [`models/hand_pose/evaluation_report.json`](file:///c:/Users/aarun/OneDrive/Desktop/sign%202/models/hand_pose/evaluation_report.json)

---

## 4. Live UI & Integration

- **Single Camera Interface:** Reuses the existing Gradio webcam input component without extra OpenCV popups.
- **Visual Landmark Feedback:** Draws green landmark circles (radius 5) at all 21 joints and blue finger connection lines directly on the live camera stream.
- **Hand Tracking & Handshape Telemetry:**
  ```text
  Status: Hand detected (1 hand)
  • Left hand tracked: NO
  • Right hand tracked: YES
  • Handshape: Right: Open Palm
  • Landmark count: 21/21 (21/21 per hand verified)
  • Camera FPS: 28.5
  ```
- **Sentence Builder & Multilingual Translation:**
  - Auto-add verified predictions with cooldown gating to prevent repeated word injection.
  - Bottom controls: `Start Recognition`, `Pause`, `Stop & Clear`, `Undo Last Sign`, `Confirm Sentence`, `Translate Sentence`, `Speak (TTS)`.
- **Port Allocation:** Running on isolated local ports:
  - **FastAPI Backend:** `http://0.0.0.0:8001/api/docs` (Health check: `/api/health`)
  - **Gradio Frontend:** `http://0.0.0.0:7861/`

---

## 5. Verification Results

Full automated testing was executed with `pytest`:
```text
tests/test_automated_suite.py ...........                                [ 21%]
tests/test_core.py ........                                              [ 36%]
tests/test_hand_pose_pipeline.py ..........                              [ 55%]
tests/test_isl_integration.py ............                               [ 78%]
tests/test_live_pipeline_repaired.py ...........                         [100%]

====================== 52 passed, 17 warnings in 26.47s =======================
```
All **52/52 unit and integration tests passed cleanly**.

