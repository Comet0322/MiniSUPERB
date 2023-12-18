"""
Parse the commonly used corpus into standardized dictionary structure
"""

from .librilight import LibriLight
from .librispeech import LibriSpeech
from .voxceleb1sid import VoxCeleb1SID

__all__ = [
    "FluentSpeechCommands",
    "IEMOCAP",
    "LibriSpeech",
    "LibriLight",
    "Quesst14",
    "SNIPS",
    "SpeechCommandsV1",
    "VoxCeleb1SID",
    "VoxCeleb1SV",
]
