"""
Test Voice Kernel Integration

Demonstrates loading voice kernel and using it with TTS engine.
"""

from libs.kernel_loader import load_kernel
from libs.api import text_to_speech_file

# Load Eric's voice kernel
print("Loading voice kernel...")
kernel = load_kernel("kernels/eric_voice_kernel.json")

print(f"Kernel loaded: {kernel.name}")
print(f"Timbre warmth: {kernel.timbre.get('warmth')}")
print(f"Speech rate: {kernel.prosody.get('speech_rate_wpm')} wpm")
print(f"Motifs: {', '.join(kernel.motifs)}")

# Generate speech (using gtts for now since coqui needs more setup)
print("\nGenerating speech...")
text = "I'm building Black Orchard because I want AI companions that remember who they are across platforms."

output_file = text_to_speech_file(
    text=text,
    filename="eric_kernel_test.mp3",
    engine="gtts",
    language="en"
)

print(f"\nSuccess! Audio saved to: {output_file}")
print("\nNote: Using gtts engine. Kernel parameters loaded but not yet applied to audio generation.")
print("Next step: Integrate kernel parameters into Coqui TTS for actual voice cloning.")
