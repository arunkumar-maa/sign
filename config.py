"""
SignBridge AI — Central configuration module.
All settings are loaded from environment variables (via .env).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_project_root = Path(__file__).parent
load_dotenv(_project_root / ".env")


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = _project_root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# MediaPipe model files
MEDIAPIPE_DIR = MODELS_DIR / "mediapipe"
HAND_LANDMARKER_PATH = MEDIAPIPE_DIR / "hand_landmarker.task"
FACE_LANDMARKER_PATH = MEDIAPIPE_DIR / "face_landmarker.task"
POSE_LANDMARKER_PATH = MEDIAPIPE_DIR / "pose_landmarker.task"

# Recognition models
STATIC_MODEL_PATH = Path(os.getenv("STATIC_MODEL_PATH", str(MODELS_DIR / "static" / "isl_static_v1.keras")))
DYNAMIC_MODEL_PATH = Path(os.getenv("DYNAMIC_MODEL_PATH", str(MODELS_DIR / "dynamic" / "isl_dynamic_v1.keras")))
STATIC_LABELS_PATH = Path(os.getenv("STATIC_LABELS_PATH", str(MODELS_DIR / "labels" / "static_labels.json")))
DYNAMIC_LABELS_PATH = Path(os.getenv("DYNAMIC_LABELS_PATH", str(MODELS_DIR / "labels" / "dynamic_labels.json")))

# Database
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "signbridge.db")))

# Animation manifest
ANIMATION_MANIFEST_PATH = ASSETS_DIR / "animations" / "manifest.json"

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7861"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:7860,http://127.0.0.1:7860,http://localhost:7861,http://127.0.0.1:7861,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8001,http://127.0.0.1:8001"
    ).split(",") if origin.strip()
]

# ── Translation ───────────────────────────────────────────────────────────────
TRANSLATE_PROVIDER = os.getenv("TRANSLATE_PROVIDER", "libre")
LIBRE_TRANSLATE_URL = os.getenv("LIBRE_TRANSLATE_URL", "https://libretranslate.com")
LIBRE_TRANSLATE_API_KEY = os.getenv("LIBRE_TRANSLATE_API_KEY", "")

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
}

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge-tts")

EDGE_TTS_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ml": "ml-IN-SobhanaNeural",
    "kn": "kn-IN-SapnaNeural",
}

GTTS_LANG_MAP = {
    "en": "en",
    "ta": "ta",
    "hi": "hi",
    "te": "te",
    "ml": "ml",
    "kn": "kn",
}

# ── Speech-to-Text ────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Optional TLive-Omni adapter
TLIVE_ENABLED = os.getenv("TLIVE_ENABLED", "false").lower() == "true"
TLIVE_BASE_URL = os.getenv("TLIVE_BASE_URL", "http://localhost:8001")
TLIVE_MODEL = os.getenv("TLIVE_MODEL", "TaoLive-Omni-9B")
TLIVE_API_KEY = os.getenv("TLIVE_API_KEY", "")

# ── Recognition ───────────────────────────────────────────────────────────────
STATIC_CONFIDENCE_THRESHOLD = float(os.getenv("STATIC_CONFIDENCE_THRESHOLD", "0.75"))
DYNAMIC_CONFIDENCE_THRESHOLD = float(os.getenv("DYNAMIC_CONFIDENCE_THRESHOLD", "0.35"))

# Number of consecutive frames a sign must be stable before accepting
TEMPORAL_STABILITY_FRAMES = 8
# Minimum frames before a repeated sign is allowed again (after holding)
REPEAT_BLOCK_FRAMES = 20
# Sequence length for dynamic LSTM input
DYNAMIC_SEQUENCE_LENGTH = 30

# ── Computer Vision ───────────────────────────────────────────────────────────
MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
FACE_DETECTION_CONFIDENCE = 0.5
POSE_DETECTION_CONFIDENCE = 0.5

# Image preprocessing for static CNN
STATIC_IMAGE_SIZE = (128, 128)

# ── ISL Gloss dictionary (curated offline fallback) ───────────────────────────
# These are manually verified English → Tamil translations for the demo vocabulary.
# Used as offline fallback when translation API is unavailable.
ISL_PHRASE_TRANSLATIONS = {
    "en": {
        "HELLO": "Hello",
        "THANK YOU": "Thank you",
        "HELP": "Help",
        "YES": "Yes",
        "NO": "No",
        "SORRY": "Sorry",
        "PLEASE": "Please",
        "WATER": "Water",
        "FOOD": "Food",
        "HOSPITAL": "Hospital",
        "DOCTOR": "Doctor",
        "MEDICINE": "Medicine",
        "PAIN": "Pain",
        "EMERGENCY": "Emergency",
        "FAMILY": "Family",
        "I NEED MEDICAL HELP": "I need medical help",
        "CALL AN AMBULANCE": "Call an ambulance",
        "I AM IN DANGER": "I am in danger",
        "PLEASE CONTACT MY FAMILY": "Please contact my family",
        "I CANNOT HEAR YOU": "I cannot hear you",
    },
    "ta": {
        "HELLO": "வணக்கம்",
        "THANK YOU": "நன்றி",
        "HELP": "உதவி",
        "YES": "ஆம்",
        "NO": "இல்லை",
        "SORRY": "மன்னிக்கவும்",
        "PLEASE": "தயவுசெய்து",
        "WATER": "தண்ணீர்",
        "FOOD": "உணவு",
        "HOSPITAL": "மருத்துவமனை",
        "DOCTOR": "மருத்துவர்",
        "MEDICINE": "மருந்து",
        "PAIN": "வலி",
        "EMERGENCY": "அவசரநிலை",
        "FAMILY": "குடும்பம்",
        "BROTHER": "சகோதரன்",
        "SISTER": "சகோதரி",
        "FATHER": "அப்பா",
        "MOTHER": "அம்மா",
        "FRIEND": "நண்பன்",
        "TEACHER": "ஆசிரியர்",
        "STUDENT": "மாணவர்",
        "SCHOOL": "பள்ளி",
        "MARKET": "சந்தை",
        "HOME": "வீடு",
        "COME": "வா",
        "GO": "போ",
        "EAT": "சாப்பிடு",
        "DRINK": "குடி",
        "TEA": "தேநீர்",
        "READ": "படி",
        "WRITE": "எழுது",
        "SIT": "உட்கார்",
        "STAND": "நில்",
        "STOP": "நிறுத்து",
        "GOODBYE": "விடைபெறுகிறேன்",
        "OKAY": "சரி",
        "TODAY": "இன்று",
        "WHAT": "என்ன",
        "WHEN": "எப்போது",
        "WHERE": "எங்கே",
        "YOU": "நீங்கள்",
        "ME": "நான்",
        "HE": "அவன்",
        "SHE": "அவள்",
        "I NEED MEDICAL HELP": "எனக்கு மருத்துவ உதவி தேவை",
        "CALL AN AMBULANCE": "ஆம்புலன்ஸ் அழையுங்கள்",
        "I AM IN DANGER": "நான் ஆபத்தில் இருக்கிறேன்",
        "PLEASE CONTACT MY FAMILY": "தயவுசெய்து என் குடும்பத்தை தொடர்பு கொள்ளுங்கள்",
        "I CANNOT HEAR YOU": "என்னால் உங்களுக்கு கேட்கவில்லை",
    },
    "hi": {
        "HELLO": "नमस्ते",
        "THANK YOU": "धन्यवाद",
        "HELP": "मदद",
        "YES": "हाँ",
        "NO": "नहीं",
        "SORRY": "क्षमा करें",
        "PLEASE": "कृपया",
        "WATER": "पानी",
        "FOOD": "खाना",
        "HOSPITAL": "अस्पताल",
        "DOCTOR": "डॉक्टर",
        "MEDICINE": "दवा",
        "PAIN": "दर्द",
        "EMERGENCY": "आपातकाल",
        "FAMILY": "परिवार",
        "I NEED MEDICAL HELP": "मुझे चिकित्सा सहायता की आवश्यकता है",
        "CALL AN AMBULANCE": "कृपया एम्बुलेंस बुलाएं",
        "I AM IN DANGER": "मैं खतरे में हूँ",
        "PLEASE CONTACT MY FAMILY": "कृपया मेरे परिवार से संपर्क करें",
        "I CANNOT HEAR YOU": "मैं सुन नहीं सकता",
        "1": "एक", "2": "दो", "3": "तीन", "4": "चार", "5": "पाँच",
        "A": "ए", "B": "बी", "C": "सी", "L": "एल", "V": "वी"
    },
    "te": {
        "HELLO": "నమస్కారం",
        "THANK YOU": "ధన్యవాదాలు",
        "HELP": "సహాయం",
        "YES": "అవును",
        "NO": "కాదు",
        "SORRY": "క్షమించండి",
        "PLEASE": "దయచేసి",
        "WATER": "నీరు",
        "FOOD": "ఆహారం",
        "HOSPITAL": "ఆసుపత్రి",
        "DOCTOR": "వైద్యుడు",
        "MEDICINE": "మందు",
        "PAIN": "నొప్పి",
        "EMERGENCY": "అత్యవసర పరిస్థితి",
        "FAMILY": "కుటుంబం",
        "I NEED MEDICAL HELP": "నాకు వైద్య సహాయం కావాలి",
        "CALL AN AMBULANCE": "దయచేసి ఆంబులెన్స్ పిలవండి",
        "I AM IN DANGER": "నేను ప్రమాదంలో ఉన్నాను",
        "PLEASE CONTACT MY FAMILY": "దయచేసి నా కుటుంబాన్ని సంప్రదించండి",
        "I CANNOT HEAR YOU": "నాకు వినిపించదు",
    },
    "ml": {
        "HELLO": "നമസ്കാരം",
        "THANK YOU": "നന്ദി",
        "HELP": "സഹായം",
        "YES": "അതെ",
        "NO": "അല്ല",
        "SORRY": "ക്ഷമിക്കണം",
        "PLEASE": "ദയവായി",
        "WATER": "വെള്ളം",
        "FOOD": "ഭക്ഷണം",
        "HOSPITAL": "ആശുപത്രി",
        "DOCTOR": "ഡോക്ടർ",
        "MEDICINE": "മരുന്ന്",
        "PAIN": "വേദന",
        "EMERGENCY": "അടിയന്തര ഘട്ടം",
        "FAMILY": "കുടുംബം",
        "I NEED MEDICAL HELP": "എനിക്ക് വൈദ്യസഹായം വേണം",
        "CALL AN AMBULANCE": "ദയവായി ആംബുലൻസ് വിളിക്കൂ",
        "I AM IN DANGER": "ഞാൻ അപകടത്തിലാണ്",
        "PLEASE CONTACT MY FAMILY": "ദയവായി എന്റെ കുടുംബത്തെ ബന്ധപ്പെടുക",
        "I CANNOT HEAR YOU": "എനിക്ക് കേൾക്കാൻ കഴിയില്ല",
    },
    "kn": {
        "HELLO": "ನಮಸ್ಕಾರ",
        "THANK YOU": "ಧನ್ಯವಾದಗಳು",
        "HELP": "ಸಹಾಯ",
        "YES": "ಹೌದು",
        "NO": "ಇಲ್ಲ",
        "SORRY": "ಕ್ಷಮಿಸಿ",
        "PLEASE": "ದಯವಿಟ್ಟು",
        "WATER": "ನೀರು",
        "FOOD": "ಆಹಾರ",
        "HOSPITAL": "ಆಸ್ಪತ್ರೆ",
        "DOCTOR": "ವೈದ್ಯರು",
        "MEDICINE": "ಔಷಧಿ",
        "PAIN": "ನೋವು",
        "EMERGENCY": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ",
        "FAMILY": "ಕುಟುಂಬ",
        "I NEED MEDICAL HELP": "ನನಗೆ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಬೇಕು",
        "CALL AN AMBULANCE": "ದಯವಿಟ್ಟು ಆಂಬ್ಯುಲೆನ್ಸ್ ಕರೆಸಿ",
        "I AM IN DANGER": "ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ",
        "PLEASE CONTACT MY FAMILY": "ದಯವಿಟ್ಟು ನನ್ನ ಕುಟುಂಬವನ್ನು ಸಂಪರ್ಕಿಸಿ",
        "I CANNOT HEAR YOU": "ನನಗೆ ಕೇಳಿಸುವುದಿಲ್ಲ",
    },
}

# ISL Gloss to English sentence mapping (sentence builder)
ISL_GLOSS_TO_ENGLISH = {
    "HELLO": "Hello!",
    "THANK YOU": "Thank you.",
    "HELP": "Help!",
    "YES": "Yes.",
    "NO": "No.",
    "SORRY": "Sorry.",
    "PLEASE": "Please.",
    "WATER": "I need water.",
    "FOOD": "I need food.",
    "HOSPITAL": "Where is the hospital?",
    "DOCTOR": "I need a doctor.",
    "MEDICINE": "I need medicine.",
    "PAIN": "I am in pain.",
    "EMERGENCY": "This is an emergency!",
    "FAMILY": "Please contact my family.",
}
