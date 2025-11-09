"""
Voice Kernel - Say Anything

Interactive voice cloning: type text, hear it in your voice!
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# Add safe globals for PyTorch 2.6+
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

from libs.kernel_loader import load_kernel
from TTS.api import TTS
import soundfile as sf

# Patch TTS audio loading
def load_audio_patch(audiopath, load_sr=22050):
    """Load audio using soundfile instead of torchaudio."""
    audio, sr = sf.read(audiopath)
    audio_tensor = torch.FloatTensor(audio)
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=1)
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    return audio_tensor

import TTS.tts.models.xtts as xtts_module
xtts_module.load_audio = load_audio_patch

print("=" * 60)
print("VOICE KERNEL - SAY ANYTHING")
print("=" * 60)

# Load kernel
print("\nLoading voice kernel...")
kernel = load_kernel("kernels/eric_voice_kernel.json", 
                     audio_path="samples/eric_voice_clean.wav")
print(f"✓ Loaded: {kernel.name}")

# Initialize TTS (only once)
print("\nInitializing Coqui TTS...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
print("✓ Ready!")

# Interactive loop
counter = 1
while True:
    print("\n" + "=" * 60)
    text = input("What should I say? (or 'quit' to exit): ")
    
    if text.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    if not text.strip():
        print("(Please enter some text)")
        continue
    
    print(f"\nGenerating speech...")
    output_file = f"output_{counter:03d}.wav"
    
    try:
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=kernel.get_speaker_wav(),
            language="en"
        )
        
        print(f"✓ Saved to: {output_file}")
        print(f"  (Play it with: start {output_file})")
        counter += 1
        
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 60)
