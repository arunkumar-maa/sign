"""About tab."""
import gradio as gr


def build_about_tab():
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
## About SignBridge AI

### Problem Statement
Over 5 million people in India use Indian Sign Language (ISL) as their primary means of communication.
However, ISL remains severely under-resourced in AI and natural language processing.
Most hearing individuals cannot understand ISL, creating a profound communication barrier in
everyday life, medical settings, and emergency situations.

### Our Solution
SignBridge AI is a Python-based accessibility application that:
- Recognizes ISL signs in real time using a webcam and trained AI models
- Converts signs to English text and Tamil/other language speech
- Allows hearing individuals to communicate back via text or voice, with ISL gloss lookup
- Provides emergency communication tools with pre-verified multilingual messages

### Technical Architecture

```
Webcam → OpenCV frames
       → MediaPipe Hand Landmarker (21 landmarks × 2 hands)
       → MediaPipe Face Landmarker (facial expression features)
       → MediaPipe Pose Landmarker (upper-body context)
       → Feature Extraction (normalized, scale-invariant)
       → Static MLP / Dynamic LSTM inference (TensorFlow/Keras)
       → Temporal Filter (stability + repeat blocking)
       → Sentence Builder (ISL gloss → English mapping)
       → LibreTranslate API (English → Tamil/other)
       → Edge-TTS / gTTS (Tamil/English audio)
       → Gradio Blocks UI (Python-only, no JavaScript)
       → FastAPI REST backend (Pydantic, auto-docs)
       → SQLite (conversation history, consent-gated)
```

### AI Ethics and Transparency
- All recognition metrics reported come from held-out test sets only
- No ISL gestures are invented or fabricated
- Avatar animations are shown only for verified, documented signs
- ASL data is strictly kept separate from ISL training data
- Emergency message translations are manually curated, not auto-translated

### Datasets Used
- Amrita SLR Real-Time ISL Dataset (CC BY 4.0) — pending download
- Common ISL Phrases (CC BY 4.0) — pending download
- ISL Hand Gesture Dataset (CC BY 4.0) — pending download

### Limitations
- Demo vocabulary is limited to signs with verified dataset coverage
- Dynamic gesture model requires ISL video dataset download and training
- TTS requires internet connection (edge-tts / gTTS)
- Avatar animations are in development; most signs show "unsupported" status during hackathon

### Future Enhancements
- Expanded vocabulary (500+ ISL signs)
- Signer-independent training with more diverse datasets
- Offline TTS with a local Tamil TTS engine
- Blender-rigged 3D avatar with full finger articulation
- Mobile application
- Real-time multi-person signing detection

### References and Attribution
- Reference Architecture: [mnehru/Indian-Sign-Language-to-Text-Speech-translation](https://github.com/mnehru/Indian-Sign-Language-to-Text-Speech-translation) (MIT)
- STT Architecture Study: [TaoLiveAIGC/TLive-Omni](https://github.com/TaoLiveAIGC/TLive-Omni) (Apache-2.0)
- ISL Dataset: [ayesha-hannure/Indian-Sign-Language-dataset](https://github.com/ayesha-hannure/Indian-Sign-Language-dataset) (Apache-2.0)
- MediaPipe: Google LLC (Apache-2.0)
- Gradio: Hugging Face (Apache-2.0)
        """)

