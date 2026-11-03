"""ISL Learning tab."""

import gradio as gr


def build_learning_tab():
    with gr.Tab("📚 ISL Learning"):
        gr.Markdown("""
## ISL Learning Module
Explore Indian Sign Language signs with reference images and practice with your webcam.
*Feedback is based on actual landmark detection results — no fake similarity scores.*
        """)

        with gr.Tabs():
            with gr.Tab("🔤 Alphabet (A–Z)"):
                gr.Markdown("""
### ISL Alphabet
*Reference images will appear here once the ISL Alphabet dataset is downloaded and processed.*

**Status:** Dataset download pending. See `data/dataset_manifest.json` for download instructions.

| Letter | Status |
|--------|--------|
| A–Z | Pending dataset download |
                """)

            with gr.Tab("👋 Greetings"):
                gr.Markdown("""
### ISL Greetings

| Sign | Meaning | Notes |
|------|---------|-------|
| HELLO | Hello / Greetings | Open palm wave |
| THANK YOU | Thank you | Fingertips to chin, move outward |
| SORRY | Sorry / Apology | Fist on chest, circular motion |
| PLEASE | Please | Flat hand on chest, circular motion |

*These are general descriptions for learning reference only. Actual ISL should be learned from certified sources.*
                """)

            with gr.Tab("🆘 Emergency Signs"):
                gr.Markdown("""
### Emergency ISL Signs

| Sign | English Meaning | Tamil |
|------|----------------|-------|
| HELP | Help! | உதவி! |
| HOSPITAL | Hospital | மருத்துவமனை |
| DOCTOR | Doctor | மருத்துவர் |
| PAIN | I am in pain | வலி |
| EMERGENCY | Emergency | அவசரநிலை |

*Practice these signs and use the Emergency SOS tab for pre-built messages.*
                """)

            with gr.Tab("🎯 Webcam Practice"):
                gr.Markdown("""
### Webcam Practice
*Use the Live Translator tab for real-time recognition feedback.*

**Instructions:**
1. Go to the **Live Translator** tab
2. Click **Start Recognition**
3. Show a sign from the vocabulary above
4. The system will attempt to recognize it and show confidence

**Important:** Recognition feedback is only accurate for signs that the model has been trained on.
If a model is not loaded, you will see: *"Recognition model is unavailable."*
                """)

