"""
Voice Kernel Loader

Loads voice kernel JSON and provides speaker configuration for TTS engines.
Part of Black Orchard Soul Forge - Voice Kernel Architecture
"""

import json
import os
from typing import Dict, Optional
from pathlib import Path


class VoiceKernel:
    """Voice kernel container with speaker configuration."""
    
    def __init__(self, kernel_data: dict, audio_path: Optional[str] = None):
        self.data = kernel_data
        self.audio_path = audio_path
        
    @property
    def name(self) -> str:
        return self.data.get("kernel_name", "unknown")
    
    @property
    def timbre(self) -> dict:
        return self.data.get("timbre", {})
    
    @property
    def prosody(self) -> dict:
        return self.data.get("prosody", {})
    
    @property
    def motifs(self) -> list:
        return self.data.get("motifs", [])
    
    def get_speaker_wav(self) -> Optional[str]:
        """Get path to speaker reference audio."""
        return self.audio_path


def load_kernel(kernel_path: str, audio_path: Optional[str] = None) -> VoiceKernel:
    """
    Load voice kernel from JSON file.
    
    Args:
        kernel_path: Path to kernel JSON file
        audio_path: Optional path to reference audio (WAV format)
        
    Returns:
        VoiceKernel object
        
    Raises:
        FileNotFoundError: If kernel file doesn't exist
        json.JSONDecodeError: If kernel file is invalid JSON
    """
    kernel_path = Path(kernel_path)
    
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
    
    with open(kernel_path, 'r') as f:
        kernel_data = json.load(f)
    
    return VoiceKernel(kernel_data, audio_path)


def save_kernel(kernel: VoiceKernel, output_path: str):
    """Save voice kernel to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(kernel.data, f, indent=2)
