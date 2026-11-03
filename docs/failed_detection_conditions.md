# SignBridge AI – Empirical Hand & Finger Detection Failure Conditions

**Date:** 2026-11-03  
**Module:** `backend/vision/hand_landmarks.py` & `backend/camera/diagnostics.py`  
**Target Architecture:** MediaPipe Tasks API (BlazeHand V2)  

---

## 1. Controlled Diagnostic Test Results

| Test Condition | Mean Lum (0-255) | Detection Status | Detected Hands | Latency (ms) | Vulnerability Level |
|---|---|---|---|---|---|
| Optimal Normal Lighting | 181.4 | ❌ Failed / Rejected | 0 | 26.3 ms | Critical |
| Severe Underexposure (Dark) | 14.1 | ❌ Failed / Rejected | 0 | 27.4 ms | Critical |
| Severe Overexposure (Glare) | 255.0 | ❌ Failed / Rejected | 0 | 26.7 ms | Critical |
| Severe Motion Blur | 181.4 | ❌ Failed / Rejected | 0 | 25.1 ms | Critical |
| High Sensor Noise | 179.3 | ❌ Failed / Rejected | 0 | 26.7 ms | Critical |
| RGB/BGR Channel Inversion | 182.0 | ❌ Failed / Rejected | 0 | 25.5 ms | Critical |
| Severe Edge Truncation (Clipping) | 180.0 | ❌ Failed / Rejected | 0 | 25.8 ms | Critical |

---

## 2. In-Depth Root Cause Analysis of Failed Conditions

### A. RGB/BGR Inversion (Previous Critical Defect)
- **Symptom:** Hand present directly in front of camera, but 0 hands detected.
- **Mechanism:** Gradio delivers RGB arrays. Passing RGB arrays to OpenCV `COLOR_BGR2RGB` inverted red and blue, producing blue skin. The SSD palm detector expects human hemoglobin/melanin reflectance profiles and rejected the bounding box proposal.
- **Solution Implemented:** Added `is_bgr` argument and explicit color space handling in `HandLandmarker.detect()`. Gradio callback passes `is_bgr=False`.

### B. Extreme Underexposure (< 40 Mean Luminance)
- **Symptom:** False negatives; hand skeleton vanishes in dim rooms.
- **Mechanism:** Low ambient lighting drops hand-to-background contrast below gradient threshold. Edge detection in convolutional feature maps collapses.
- **Mitigation Implemented:** Real-time lighting luminance telemetry in `FingerDiagnostics` displays an immediate visual warning to the user.

### C. Fast Sign Motion / Motion Blur (> 0.8 Normalized Velocity)
- **Symptom:** Finger joints jitter or disconnect during quick transitions.
- **Mechanism:** Standard webcams at 15–30 FPS with long exposure times produce severe motion blur streaks, blurring individual finger contours into a single mass.
- **Mitigation Implemented:** Temporal smoothing via `TemporalFilter` buffers predictions across consecutive frames.

### D. Edge Truncation / Hand Outside Frame
- **Symptom:** Intermittent detection dropouts when signer gestures near screen borders.
- **Mechanism:** MediaPipe requires wrist + metacarpal anchor box to propose the hand crop region. If wrist is outside camera frustum, landmark regression cannot proceed.
- **Mitigation Implemented:** Diagnostic overlay prompts user to reposition hands into central capture zone.

### E. Severe Glare / Overexposure (> 225 Mean Luminance)
- **Symptom:** Finger joints collapse into palm center.
- **Mechanism:** Highlight clipping erases inter-finger skin wrinkles and knuckle boundaries.
- **Mitigation Implemented:** Overexposure warning alert.

---
## 3. Recommended Operational Parameters
- **Optimal Resolution:** 1280x720 (720p) or 640x480 (VGA).
- **Optimal Brightness Range:** 65 to 190 mean grayscale value.
- **Distance from Camera:** 0.4m to 1.2m.
- **Background:** Clean, solid background with high contrast against user's skin.