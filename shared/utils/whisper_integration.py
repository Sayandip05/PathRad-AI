"""
OpenAI Whisper Local Integration
Multilingual speech recognition for voice-based patient input
"""

import whisper
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperManager:
    """
    Manages local Whisper model for multilingual speech recognition
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model

        Args:
            model_size: Model size - "tiny", "base", "small", "medium", "large"
                      base = 74M parameters, ~1GB VRAM
                      small = 244M parameters, ~2GB VRAM
        """
        self.model_size = model_size
        self.model = None
        self.device = None

    def load_model(self) -> bool:
        """
        Load Whisper model on GPU if available
        """
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")

            # Check device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.warning("GPU not available, using CPU for Whisper")

            # Load model
            self.model = whisper.load_model(self.model_size).to(self.device)

            logger.info(
                f"✅ Whisper {self.model_size} model loaded successfully on {self.device}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe(
        self, audio_path: str, language: Optional[str] = None, task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)
            language: Language code (e.g., 'en', 'hi', 'es', 'fr', 'bn')
                     If None, auto-detect language
            task: "transcribe" or "translate" (to English)

        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                return {
                    "success": False,
                    "error": "Failed to load Whisper model",
                    "text": "",
                }

        try:
            logger.info(f"Transcribing audio: {audio_path}")

            # Transcribe
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=torch.cuda.is_available(),  # Use FP16 on GPU
            )

            transcription = {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "duration": result.get("duration", 0),
            }

            logger.info(
                f"Transcription complete. Language: {transcription['language']}, Text length: {len(transcription['text'])}"
            )
            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"success": False, "error": str(e), "text": ""}

    def transcribe_for_patient_input(
        self, audio_path: str, input_type: str = "chief_complaint"
    ) -> Dict[str, Any]:
        """
        Transcribe patient audio input with medical context

        Args:
            audio_path: Path to audio file
            input_type: Type of input - "chief_complaint", "clinical_history", "symptoms"

        Returns:
            Structured transcription with medical context
        """
        result = self.transcribe(audio_path)

        if result["success"]:
            # Add medical context based on input type
            result["input_type"] = input_type
            result["medical_context"] = self._extract_medical_entities(result["text"])

        return result

    def _extract_medical_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract medical entities from transcribed text
        Simple keyword-based extraction
        """
        text_lower = text.lower()

        entities = {"symptoms": [], "duration": [], "severity": [], "body_parts": []}

        # Symptom keywords
        symptom_keywords = [
            "pain",
            "ache",
            "fever",
            "cough",
            "breathing",
            "nausea",
            "vomiting",
            "diarrhea",
            "headache",
            "dizziness",
            "fatigue",
            "weakness",
            "chest pain",
            "shortness of breath",
            "wheezing",
            "blood",
            "swelling",
            "rash",
            "itching",
            "numbness",
        ]

        for symptom in symptom_keywords:
            if symptom in text_lower:
                entities["symptoms"].append(symptom)

        # Duration keywords
        duration_keywords = [
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
            "hour",
            "hours",
            "minute",
            "minutes",
        ]

        for duration in duration_keywords:
            if duration in text_lower:
                entities["duration"].append(duration)

        # Severity keywords
        severity_keywords = [
            "severe",
            "mild",
            "moderate",
            "intense",
            "sharp",
            "dull",
            "burning",
            "throbbing",
            "constant",
            "intermittent",
        ]

        for severity in severity_keywords:
            if severity in text_lower:
                entities["severity"].append(severity)

        return entities

    def detect_language(self, audio_path: str) -> str:
        """
        Detect language of audio file
        """
        if self.model is None:
            self.load_model()

        try:
            # Load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # Detect language
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            return detected_lang

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """
        Get list of languages supported by Whisper
        """
        return {
            "en": "English",
            "zh": "Chinese",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ko": "Korean",
            "fr": "French",
            "ja": "Japanese",
            "pt": "Portuguese",
            "tr": "Turkish",
            "pl": "Polish",
            "ca": "Catalan",
            "nl": "Dutch",
            "ar": "Arabic",
            "sv": "Swedish",
            "it": "Italian",
            "id": "Indonesian",
            "hi": "Hindi",
            "fi": "Finnish",
            "vi": "Vietnamese",
            "he": "Hebrew",
            "uk": "Ukrainian",
            "el": "Greek",
            "ms": "Malay",
            "cs": "Czech",
            "ro": "Romanian",
            "da": "Danish",
            "hu": "Hungarian",
            "ta": "Tamil",
            "no": "Norwegian",
            "th": "Thai",
            "ur": "Urdu",
            "hr": "Croatian",
            "bg": "Bulgarian",
            "lt": "Lithuanian",
            "la": "Latin",
            "mi": "Maori",
            "ml": "Malayalam",
            "cy": "Welsh",
            "sk": "Slovak",
            "te": "Telugu",
            "fa": "Persian",
            "lv": "Latvian",
            "bn": "Bengali",
            "sr": "Serbian",
            "az": "Azerbaijani",
            "sl": "Slovenian",
            "kn": "Kannada",
            "et": "Estonian",
            "mk": "Macedonian",
            "br": "Breton",
            "eu": "Basque",
            "is": "Icelandic",
            "hy": "Armenian",
            "ne": "Nepali",
            "mn": "Mongolian",
            "bs": "Bosnian",
            "kk": "Kazakh",
            "sq": "Albanian",
            "sw": "Swahili",
            "gl": "Galician",
            "mr": "Marathi",
            "pa": "Punjabi",
            "si": "Sinhala",
            "km": "Khmer",
            "sn": "Shona",
            "yo": "Yoruba",
            "so": "Somali",
            "af": "Afrikaans",
            "oc": "Occitan",
            "ka": "Georgian",
            "be": "Belarusian",
            "tg": "Tajik",
            "sd": "Sindhi",
            "gu": "Gujarati",
            "am": "Amharic",
            "yi": "Yiddish",
            "lo": "Lao",
            "uz": "Uzbek",
            "fo": "Faroese",
            "ht": "Haitian Creole",
            "ps": "Pashto",
            "tk": "Turkmen",
            "nn": "Nynorsk",
            "mt": "Maltese",
            "sa": "Sanskrit",
            "lb": "Luxembourgish",
            "my": "Myanmar",
            "bo": "Tibetan",
            "tl": "Tagalog",
            "mg": "Malagasy",
            "as": "Assamese",
            "tt": "Tatar",
            "haw": "Hawaiian",
            "ln": "Lingala",
            "ha": "Hausa",
            "ba": "Bashkir",
            "jw": "Javanese",
            "su": "Sundanese",
        }


# Global Whisper instance
whisper_manager = WhisperManager(model_size="base")


def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for audio transcription
    """
    return whisper_manager.transcribe(audio_path, language)


def process_patient_voice_input(
    audio_path: str, input_type: str = "chief_complaint"
) -> Dict[str, Any]:
    """
    Process patient voice input with medical context extraction
    """
    return whisper_manager.transcribe_for_patient_input(audio_path, input_type)
