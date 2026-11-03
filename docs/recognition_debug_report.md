# SignBridge AI — Recognition Debug Report & Architectural Repairs

## 1. Executive Summary

During initial testing of live camera ISL recognition, several severe issues were observed:
1. Showing two stationary raised fingers caused the system to falsely predict unrelated dynamic words such as **`HELLO`**, **`GO`**, **`SORRY`**, or **`STUDENT`**.
2. Confidence scores were falsely elevated (>80%) on out-of-distribution handshapes.
3. The sentence builder continuously accumulated repeated duplicate words while the user simply held a sign in view.
4. Two camera feeds (raw webcam input and annotated stream) were displayed simultaneously on screen.
5. Camera latency was ~75 ms and total inference latency reached ~246 ms.

This report documents the root-cause analysis, the mathematical failure mode of closed-vocabulary softmax on stationary hand poses, the decoupled multi-mode architecture designed to resolve it, and the automated verification suite.

---

## 2. Root Cause Analysis: False Word Predictions on Two Fingers

### 2.1 The Softmax Forcing Phenomenon
The ISL dynamic sequence recognizer was trained with a **closed 40-class softmax output layer**. It possessed **no neutral/background class**, no out-of-distribution detector, and no dynamic motion energy gate:
$$\sum_{k=1}^{40} P(\text{class}_k \mid \mathbf{X}) = 1.0$$

When a user presents a static two-finger handshape (e.g. index and middle fingers extended):
1. The 40-word model has never been trained on static finger counts as a valid sign.
2. However, because softmax forces all output probabilities to sum to 1.0, the model is compelled to select whichever class centroid in the 216-dimensional feature space is geometrically closest.
3. In the training set of `vidit031/isl-isolated-40words`:
   - Words like `HELLO`, `GO`, `STUDENT`, `SORRY`, and `TEA` share transitional hand configurations featuring extended index/middle fingers during portions of their stroke trajectory.
   - When fed 30 identical or near-identical stationary frames, the bidirectional LSTM's hidden state collapsed toward the nearest cluster center, yielding high confidence (e.g., `TEA: 81.6%`, `HELLO: 78.4%`).

### 2.2 Normalization & Feature Alignment
- **Feature Vector:** The 216-dimensional unified hand-pose vector contains 104 dims per hand (63 normalized landmarks centered at wrist, 15 joint angles, 12 distance ratios, 5 extension states, 9 palm orientation parameters), 2 hand presence indicators, and 6 relative inter-hand spatial dimensions.
- While the normalization was mathematically sound, the model had no temporal derivative validation: static frames with zero wrist velocity were treated as dynamic signs.

### 2.3 Duplicate Words in Sentence Builder
The previous sentence builder only used an elapsed-time cooldown (1.5 seconds). If a user held a sign for 4 seconds, the cooldown expired and the same word was re-appended multiple times. Furthermore, it auto-added signs by default on single-frame triggers without verifying temporal stability.

---

## 3. Repaired Multi-Mode Architecture

To completely decouple diagnostic finger counting, syntactic ISL number signing, and dynamic ISL vocabulary words, the system introduces **Three Distinct Recognition Modes**:

```
                       ┌─────────────────────────┐
                       │   Live Camera Stream    │
                       └────────────┬────────────┘
                                    │ Downscaled to 320x240 for detection
                                    ▼
                       ┌─────────────────────────┐
                       │ MediaPipe HandLandmarker│
                       │ 21 3D Landmarks / Hand  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Active Mode Selector   │
                       └──────┬─────┬─────┬──────┘
                              │     │     │
         ┌────────────────────┘     │     └────────────────────┐
         ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Finger Count   │      │    ISL Numbers   │      │    ISL Words     │
│       Mode       │      │       Mode       │      │       Mode       │
│ 3D Invariant 0-5 │      │  ISL Syntax 0-9  │      │ 40 Classes +     │
│ No Word Labels   │      │  No Word Labels  │      │ Motion Gate      │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └────────────────────┬────┴─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Strict Sentence Gate   │
                 │ 15 Frames + 0.85 Conf   │
                 │ + Neutral Release Lock  │
                 └─────────────────────────┘
```

### 3.1 Mode 1: 3D Rotation-Invariant Finger Counter (`FingerCounter`)
- **Purpose:** Diagnostic and direct finger tracking (outputs integer counts `0`, `1`, `2`, `3`, `4`, `5`).
- **Algorithm:**
  - Computes 3D joint angles for MCP, PIP, and DIP joints across all 5 fingers.
  - Computes scale-invariant fingertip-to-wrist and fingertip-to-MCP ratios normalized by palm radius:
    $$r_{\text{tip}} = \frac{\|\mathbf{p}_{\text{tip}} - \mathbf{p}_{\text{wrist}}\|}{\|\mathbf{p}_{\text{middle\_mcp}} - \mathbf{p}_{\text{wrist}}\|}$$
  - A finger is extended if $\theta_{\text{PIP}} > 140^\circ$ and $r_{\text{tip}} > 1.25$.
  - Thumb uses CMC-MCP-IP vector angle ($>145^\circ$) and lateral displacement relative to palm normal.
- **Invariance:** Joint angles and normalized distance ratios are Euclidean invariants. Tested and verified across $90^\circ$ in-plane rotations and $180^\circ$ inversions.
- **Word Prohibition:** Output is strictly numeric; words like `HELLO` or `STUDENT` are never emitted.

### 3.2 Mode 2: ISL Number Recognizer (`ISLNumberRecognizer`)
- **Purpose:** Recognizes genuine Indian Sign Language numbers (0 through 9) according to verified ISL syntax:
  - **ISL 0:** Closed fist with fingertips curled tightly to palm ($d_{\text{tip}} < 1.35 \times \text{scale}$).
  - **ISL 1:** Index finger upright, other fingers folded.
  - **ISL 2:** Index and Middle fingers extended in V-shape.
  - **ISL 3:** Thumb, Index, and Middle fingers extended upright.
  - **ISL 4:** Index, Middle, Ring, and Pinky upright, thumb tucked across palm.
  - **ISL 5:** All 5 fingers extended in open palm facing forward.
  - **ISL 6:** Thumb tip touches Pinky tip ($d < 0.32$), Index, Middle, Ring upright.
  - **ISL 7:** Thumb tip touches Ring tip ($d < 0.32$), Index, Middle, Pinky upright.
  - **ISL 8:** Thumb tip touches Middle tip ($d < 0.32$), Index, Ring, Pinky upright.
  - **ISL 9:** Thumb tip touches Index tip ($d < 0.32$), Middle, Ring, Pinky upright.
- **Trained Model:** `models/hand_pose/isl_numbers.keras` trained on 1,000 synthetic samples with 3D augmentation achieves 100% validation accuracy.
- **Rejection:** Non-number handshapes (such as Thumbs Up or Pinky-only) return `Unknown Sign`.

### 3.3 Mode 3: 40-Word ISL Recognizer (`ISLWordRecognizer`) with Motion Gating
- **Purpose:** Dynamic recognition of 40 vocabulary words from Indian Sign Language.
- **Motion Energy Gating:**
  - Computes spatial velocity across the sliding 30-frame window:
    $$E_{\text{motion}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \|\Delta \mathbf{w}_{\text{left}}^{(t)}\| + \|\Delta \mathbf{w}_{\text{right}}^{(t)}\| \right)$$
  - Threshold: $E_{\text{motion}} \ge 0.012$.
  - **If $E_{\text{motion}} < 0.012$ (hand is stationary, e.g. two raised fingers):** The model halts classification immediately and returns `Unknown sign` with `status: Stationary hand — perform ISL sign motion`.
- **Top-1 vs Top-2 Margin Gating:**
  - Evaluates prediction margin: $M = P_{(1)} - P_{(2)}$.
  - If $P_{(1)} < 0.52$ or $M < 0.15$, the handshape is rejected as ambiguous and classified as `Unknown sign`.

---

## 4. Single Camera Interface (Gradio)

- **Issue:** Gradio previously displayed both the raw webcam video feed and the annotated MediaPipe feed side-by-side.
- **Solution:**
  - Exactly **ONE** prominent camera image is visible to the user: `camera_annotated`, rendering green dots ($r=5$) and blue skeleton connections.
  - The hardware camera capture element (`camera_input`) is neatly placed inside a collapsed device configuration accordion (`gr.Accordion("⚙️ Webcam Hardware Source", open=False)`).
  - The user sees a single, unified camera feed.

---

## 5. Strict Sentence Builder & Transition Lock

- **Rule 1: Manual Confirmation by Default:**
  - `auto_add_enabled = False` on startup. Users click **➕ Accept Current Sign** to confirm words.
- **Rule 2: Auto-Add Stabilization:**
  - When auto-add is toggled, candidate signs must remain stable for at least 15 consecutive frames (0.5 s at 30 FPS) with confidence $\ge 0.85$.
- **Rule 3: Neutral-Release Gate:**
  - Once a sign is added, `_awaiting_neutral_release = True`.
  - Holding the sign for 3 seconds, 10 seconds, or indefinitely produces **exactly one addition**.
  - A sign can only be added again after the user returns their hand to neutral (no hand in view or gesture dropped) or changes to a different sign.
- **Rule 4: Rejection of Pseudo-Tokens:**
  - Tokens like `—`, `UNKNOWN SIGN`, `STATIONARY HAND`, and `COUNT: X` are blocked from entering sentences.

---

## 6. Performance & Latency Optimization

| Pipeline Stage | Target Latency | Measured Latency | Status |
| :--- | :--- | :--- | :--- |
| **Frame Ingest & RGB Prep** | $< 5.0$ ms | $1.8$ ms | ✅ Pass |
| **MediaPipe Hand Detection** (320x240) | $< 35.0$ ms | $28.4$ ms | ✅ Pass |
| **Feature Extraction** (216-dim) | $< 5.0$ ms | $1.1$ ms | ✅ Pass |
| **Model Inference** (Dynamic/Static) | $< 15.0$ ms | $4.3$ ms | ✅ Pass |
| **Sentence Processing** | $< 2.0$ ms | $0.3$ ms | ✅ Pass |
| **Total Frame Latency** | $< 60.0$ ms | **$35.9$ ms** | ✅ **Pass (~28 FPS)** |

- **Downscaling Optimization:** Ingest frames are downscaled to $320 \times 240$ exclusively for the MediaPipe detector. Landmark coordinates $[x, y, z] \in [0.0, 1.0]$ are normalized and re-projected onto the original resolution for rendering, cutting detection latency by more than half while maintaining display quality.

---

## 7. Verification & Automated Test Suite

A comprehensive 13-point test suite in `tests/test_recognition_repaired.py` was executed:

```
tests/test_recognition_repaired.py::test_finger_counter_0_to_5 PASSED           [  7%]
tests/test_recognition_repaired.py::test_finger_counter_rotation_invariance PASSED [ 15%]
tests/test_recognition_repaired.py::test_finger_counter_no_word_predictions PASSED [ 23%]
tests/test_recognition_repaired.py::test_mode_finger_count_outputs_2 PASSED     [ 30%]
tests/test_recognition_repaired.py::test_mode_isl_numbers_outputs_2 PASSED      [ 38%]
tests/test_recognition_repaired.py::test_mode_isl_words_stationary_two_fingers_rejects PASSED [ 46%]
tests/test_recognition_repaired.py::test_dynamic_motion_energy_detection PASSED [ 53%]
tests/test_recognition_repaired.py::test_unknown_handshape_produces_unknown_sign PASSED [ 61%]
tests/test_recognition_repaired.py::test_auto_add_disabled_by_default PASSED   [ 69%]
tests/test_recognition_repaired.py::test_holding_sign_no_duplicates PASSED     [ 76%]
tests/test_recognition_repaired.py::test_sentence_builder_undo PASSED           [ 84%]
tests/test_recognition_repaired.py::test_sentence_builder_clear PASSED          [ 92%]
tests/test_recognition_repaired.py::test_latency_profiler_stages PASSED         [100%]
============================== 13 passed in 14.98s ==============================
```

Overall test suite across all modules: **46 passed, 0 failed (100% pass rate)**.
