"""
Voice Kernel + Coqui TTS Integration Test

Uses Eric's voice kernel and recorded audio for voice cloning.
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"  # Agree to Coqui TOS

import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# Add safe globals for PyTorch 2.6+
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

from libs.kernel_loader import load_kernel
from TTS.api import TTS

print("=" * 60)
print("VOICE KERNEL + COQUI TTS VOICE CLONING TEST")
print("=" * 60)

# Load Eric's voice kernel
print("\n1. Loading voice kernel...")
kernel = load_kernel("kernels/eric_voice_kernel.json", 
                     audio_path="samples/eric_voice_clean.wav")

print(f"   ✓ Kernel: {kernel.name}")
print(f"   ✓ Warmth: {kernel.timbre.get('warmth')}")
print(f"   ✓ Speech rate: {kernel.prosody.get('speech_rate_wpm')} wpm")
print(f"   ✓ Audio sample: {kernel.get_speaker_wav()}")

# Initialize Coqui TTS with XTTS model (supports voice cloning)
print("\n2. Initializing Coqui TTS (this may download the model)...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Text to synthesize
text = "I'm building Black Orchard because I want AI companions that remember who they are across platforms."

# Patch TTS audio loading to avoid torchcodec
import soundfile as sf
import torch

def load_audio_patch(audiopath, load_sr=22050):
    """Load audio using soundfile instead of torchaudio."""
    audio, sr = sf.read(audiopath)
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio)
    # Ensure mono
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=1)
    # Add batch dimension - return as [1, samples]
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    # Return ONLY tensor - TTS doesn't expect sample rate back
    return audio_tensor

# Monkey patch the load_audio function
import TTS.tts.models.xtts as xtts_module
xtts_module.load_audio = load_audio_patch

# Generate with voice cloning
print("\n3. Generating speech with your voice...")
print(f"   Text: {text[:50]}...")

output_file = "eric_voice_clone_test.wav"

tts.tts_to_file(
    text=text,
    file_path=output_file,
    speaker_wav=kernel.get_speaker_wav(),
    language="en"
)

print(f"\n✓ SUCCESS!")
print(f"✓ Voice-cloned audio saved to: {output_file}")
print(f"\nPlay the file to hear your AI voice clone!")
print("=" * 60)
