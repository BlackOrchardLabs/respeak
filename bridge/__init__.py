# Re:speak Emotional Bridge
# Connects Nemotron ASR → Voice Kernel Modulation → TTS

from .asr_client import NemotronASRClient
from .emotional_analyzer import EmotionalAnalyzer
from .modulation_mapper import ModulationMapper
from .kernel_blender import KernelBlender
from .bridge import EmotionalBridge

__all__ = [
    "NemotronASRClient",
    "EmotionalAnalyzer",
    "ModulationMapper",
    "KernelBlender",
    "EmotionalBridge"
]
