"""
Home page tab for SignBridge AI.
"""

import gradio as gr


def build_home_tab():
    """Build the Home page tab."""
    with gr.Tab("🏠 Home"):
        gr.Markdown("""
# 🤟 SignBridge AI
### *Breaking Communication Barriers with AI*

---

SignBridge AI is an open-source, Python-powered accessibility tool that bridges communication
between Indian Sign Language (ISL) users and hearing individuals using real-time computer vision,
deep learning, and multilingual speech synthesis.

---

## 🎯 What This App Does

| Direction | Workflow |
|-----------|----------|
| **ISL → Text & Speech** | Webcam → MediaPipe Landmarks → ISL Model → English → Tamil Translation → Audio |
| **Speech/Text → ISL** | Microphone/Text → Transcription → Gloss Lookup → Verified Animation |

---

## 🛠️ Core Technologies

- **Computer Vision**: OpenCV + MediaPipe (Hand, Face, Pose Landmarkers)
- **AI Recognition**: TensorFlow/Keras CNN (static) + LSTM/GRU (dynamic sequences)
- **Translation**: LibreTranslate API with offline curated phrase dictionary
- **Speech Synthesis**: Edge-TTS (Tamil, English) / gTTS fallback
- **Speech Recognition**: Faster-Whisper (small model)
- **Interface**: Gradio Blocks (Python-only, no JavaScript)
- **Backend**: FastAPI with Pydantic validation
- **Database**: SQLite conversation history

---

## 🚀 Supported Signs (Demo Vocabulary)

> **Note:** The demo vocabulary is limited to signs for which verified ISL dataset data exists.
> The vocabulary will expand as datasets are downloaded, processed, and models are trained.

Current target vocabulary: HELLO, THANK YOU, HELP, YES, NO, SORRY, PLEASE, 
WATER, FOOD, HOSPITAL, DOCTOR, MEDICINE, PAIN, EMERGENCY, FAMILY
+ ISL Alphabet (A–Z, pending dataset verification)

---

## ♿ Accessibility Mission

1 in 72 people in India is deaf. ISL has over **5 million users** but remains severely
under-resourced in AI. SignBridge AI aims to provide a practical, open-source bridge
for everyday and emergency communication.

---

## ⚠️ Honest Limitations

- Recognition accuracy depends on training data quality and quantity.
- Models shown during the hackathon are trained on limited verified ISL datasets.
- Avatar animations are shown only for signs with a verified documented source.
- All metrics reported are from held-out test sets, not training data.

---

## How to Use

1. Go to **Live Translator** → click **Start Camera** → show your ISL signs
2. Go to **Two-Way Communication** → type or speak → see the ISL gloss lookup
3. Explore **ISL Learning** to practise individual signs
4. Use **Emergency SOS** for urgent communication needs
        """)

        gr.Markdown("---\n*Navigate using the tabs above. The camera starts only when you click Start.*")

