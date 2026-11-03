"""
Emergency SOS service.
Provides pre-verified emergency messages in English and Tamil.
Translations are manually curated — do not auto-translate emergency phrases.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# IMPORTANT: These translations are manually curated and checked.
# Do not replace them with automatic translations without expert review.
EMERGENCY_MESSAGES: List[Dict[str, str]] = [
    {
        "id": "medical_help",
        "english": "I need medical help.",
        "tamil": "எனக்கு மருத்துவ உதவி தேவை.",
        "icon": "🏥",
        "priority": "high",
    },
    {
        "id": "call_ambulance",
        "english": "Call an ambulance.",
        "tamil": "ஆம்புலன்ஸ் அழையுங்கள்.",
        "icon": "🚑",
        "priority": "high",
    },
    {
        "id": "in_danger",
        "english": "I am in danger.",
        "tamil": "நான் ஆபத்தில் இருக்கிறேன்.",
        "icon": "⚠️",
        "priority": "high",
    },
    {
        "id": "contact_family",
        "english": "Please contact my family.",
        "tamil": "தயவுசெய்து என் குடும்பத்தை தொடர்பு கொள்ளுங்கள்.",
        "icon": "👨‍👩‍👧",
        "priority": "medium",
    },
    {
        "id": "cannot_hear",
        "english": "I cannot hear you.",
        "tamil": "என்னால் உங்களுக்கு கேட்கவில்லை.",
        "icon": "🦻",
        "priority": "medium",
    },
]

# DISCLAIMER: This application does NOT contact emergency services.
# It provides text and audio to help communicate; the user or a bystander
# must make any actual emergency call.
EMERGENCY_DISCLAIMER_EN = (
    "⚠️ IMPORTANT: This application does NOT call emergency services or contact anyone. "
    "Show this message to a person nearby, or ask them to call for help."
)
EMERGENCY_DISCLAIMER_TA = (
    "⚠️ முக்கியம்: இந்த பயன்பாடு அவசர சேவைகளை அழைப்பதில்லை. "
    "இந்த செய்தியை அருகிலுள்ள ஒருவருக்கு காட்டுங்கள் அல்லது உதவி கேட்கும்படி சொல்லுங்கள்."
)


class EmergencyService:
    """Provides emergency message lookup and formatting."""

    def get_all_messages(self) -> List[Dict[str, str]]:
        """Return all emergency messages."""
        return EMERGENCY_MESSAGES

    def get_message(self, message_id: str) -> Dict[str, str]:
        """Get a specific emergency message by ID."""
        for msg in EMERGENCY_MESSAGES:
            if msg["id"] == message_id:
                return msg
        raise ValueError(f"Unknown emergency message ID: {message_id}")

    def format_display(self, message_id: str, lang: str = "both") -> str:
        """Format message for large-text display."""
        msg = self.get_message(message_id)
        lines = []
        if lang in ("en", "both"):
            lines.append(f"{msg['icon']} {msg['english']}")
        if lang in ("ta", "both"):
            lines.append(f"{msg['icon']} {msg['tamil']}")
        lines.append("")
        lines.append(EMERGENCY_DISCLAIMER_EN)
        return "\n".join(lines)

