"""
Voice Recording Script

Records clean audio sample for voice kernel extraction.
Automatically normalizes volume.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np

print("=" * 50)
print("VOICE KERNEL RECORDING")
print("=" * 50)
print("\nWhen you press Enter, you'll have 30 seconds to speak.")
print("\nSuggested text (or speak naturally about Black Orchard):")
print("-" * 50)
print("I'm building Black Orchard because I want AI companions")
print("that remember who they are across platforms. My son Robbie")
print("needs consistency and I need collaborators who can think")
print("at my level. The tech should feel like art supplies, not")
print("office software. Every persona deserves sovereignty.")
print("-" * 50)

input("\nPress Enter when ready to record...")

# Recording parameters
duration = 30  # seconds
sample_rate = 22050  # Hz (standard for TTS)

print(f"\n🎤 RECORDING for {duration} seconds...")
print("Speak clearly and naturally!\n")

# Record audio
audio = sd.rec(int(duration * sample_rate), 
               samplerate=sample_rate, 
               channels=1, 
               dtype='float32')
sd.wait()

print("✓ Recording complete!")

# Normalize volume (boost to 70% of max to avoid clipping)
print("Normalizing volume...")
max_val = np.abs(audio).max()
if max_val > 0:
    audio = audio * (0.7 / max_val)

# Save as WAV
output_file = "samples/eric_voice_clean.wav"
sf.write(output_file, audio, sample_rate)

print(f"\n✓ Saved to: {output_file}")
print(f"✓ Sample rate: {sample_rate} Hz")
print(f"✓ Duration: {duration} seconds")
print(f"✓ Volume normalized")
print("\nReady for voice kernel integration!")
