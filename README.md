# Voice Kernel TTS (Soul Forge: Audio Layer)

Portable, JSON-defined voice "kernels" that drive multi-engine TTS voice cloning.

## Why

Fine-tuning is heavy and non-portable. Style transfer is fast but coarse. Voice kernels are lightweight, tunable, and portable.

## What's Inside

- **Kernel spec** (timbre/prosody/motifs/emotion)
- **Kernel loader** (`libs/kernel_loader.py`)
- **Coqui XTTS v2 demo** with reference speaker WAV
- **Record/validate scripts**
- **Interactive say-anything tool**

## Demo

Listen to [demo_snowcrash_metaverse.wav](demo_snowcrash_metaverse.wav) - a voice clone speaking about the Metaverse from Snow Crash.

## Quickstart

### Requirements
- Python 3.11 (required for Coqui TTS)
- Visual Studio Build Tools (Windows)
- 30-second clean voice sample (22,050 Hz WAV)

### Installation

```bash
# Clone the repository
git clone https://github.com/BlackOrchardLabs/re-speak.git
cd re-speak

# Install dependencies (use Python 3.11)
C:\Python311\python.exe -m pip install TTS soundfile sounddevice transformers==4.33.0
```

### Record Your Voice

```bash
C:\Python311\python.exe record_voice.py
```

This will:
- Prompt you to record 30 seconds
- Automatically normalize volume
- Save to `samples/eric_voice_clean.wav`

### Create Your Voice Kernel

Create `kernels/your_voice_kernel.json`:

```json
{
  "kernel_name": "your_voice_kernel_v0.1",
  "version": "text_proxy_v2",
  "source": "recorded_sample",
  "timbre": {
    "warmth": 0.85,
    "depth": 0.8,
    "nasality": 0.1,
    "breathiness": 0.3
  },
  "prosody": {
    "avg_pitch_hz": 142,
    "pitch_variance": 0.6,
    "speech_rate_wpm": 130,
    "pause_density": 0.28,
    "rhythmic_flow": "deliberate, measured"
  },
  "motifs": [
    "technical clarity",
    "quiet intensity",
    "builder's cadence"
  ],
  "emotional_range": {
    "calm": 0.92,
    "curiosity": 0.88,
    "excitement": 0.75
  }
}
```

### Clone Your Voice

#### Option 1: Interactive (Recommended)

```bash
C:\Python311\python.exe say_anything.py
```

Type any text and hear it in your voice!

#### Option 2: Single Test

```bash
C:\Python311\python.exe test_voice_clone.py
```

Generates `eric_voice_clone_test.wav` with predefined text.

## Kernel Example

The voice kernel format captures vocal identity as portable JSON:

```json
{
  "kernel_name": "eric_voice_kernel_v0.2",
  "timbre": {
    "warmth": 0.85,
    "depth": 0.8,
    "nasality": 0.1,
    "breathiness": 0.3
  },
  "prosody": {
    "avg_pitch_hz": 142,
    "speech_rate_wpm": 130,
    "pause_density": 0.28,
    "rhythmic_flow": "deliberate, measured"
  },
  "motifs": [
    "technical clarity",
    "quiet intensity",
    "builder's cadence",
    "subtle humor",
    "warm pauses"
  ],
  "emotional_range": {
    "calm": 0.92,
    "curiosity": 0.88,
    "excitement": 0.75,
    "frustration": 0.10
  }
}
```

## Architecture

### Voice Kernel Loader

```python
from libs.kernel_loader import load_kernel

# Load kernel with audio reference
kernel = load_kernel(
    "kernels/eric_voice_kernel.json",
    audio_path="samples/eric_voice_clean.wav"
)

# Access parameters
print(kernel.timbre)          # {"warmth": 0.85, ...}
print(kernel.prosody)         # {"avg_pitch_hz": 142, ...}
print(kernel.motifs)          # ["technical clarity", ...]
print(kernel.get_speaker_wav())  # "samples/eric_voice_clean.wav"
```

### Integration with Coqui TTS

```python
from TTS.api import TTS

# Initialize TTS
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Generate speech with voice kernel
tts.tts_to_file(
    text="Your text here",
    file_path="output.wav",
    speaker_wav=kernel.get_speaker_wav(),
    language="en"
)
```

## Technical Details

### Audio Format
- **Sample rate:** 22,050 Hz
- **Channels:** Mono (automatically converted if stereo)
- **Duration:** 30 seconds minimum recommended
- **Format:** WAV (MP3 conversion available via `convert_audio.py`)

### Dependencies
- **TTS** (Coqui XTTS v2)
- **soundfile** (audio I/O, patches torchaudio issues)
- **sounddevice** (recording)
- **transformers==4.33.0** (pinned for compatibility)
- **torch** (PyTorch 2.6+)

### Known Issues

**Windows Audio Backend:**
- Torchaudio's torchcodec dependency has compatibility issues on Windows
- We patch this by using soundfile directly (see `test_voice_clone.py`)

**PyTorch 2.6+ Security:**
- Requires safe_globals for model loading
- Automatically handled in our scripts

## Roadmap

### Immediate
- [x] Working voice clone with Coqui XTTS v2
- [x] Kernel loader system
- [x] Interactive say-anything tool
- [ ] Map kernel params → synthesis controls (rate/pitch/emotion)
- [ ] Multi-engine parity (Coqui, Piper)

### Short-term
- [ ] Automated "voice → kernel" extractor (TUSK-style)
- [ ] UI for record → extract → test workflow
- [ ] Real-time parameter tuning
- [ ] Kernel blending (70% voice A + 30% voice B)

### Long-term
- [ ] re-speak app
- [ ] Mobile deployment
- [ ] Cross-platform persona system
- [ ] Soul Forge integration (visual + voice + text + memory)

## Philosophy

Voice kernels are part of **Soul Forge**, a unified architecture for portable digital persona preservation.

---

### The Kernel Paradigm

A kernel is compressed emotional topology — not raw data, but *meaningful state*. Small enough to fit anywhere, dense enough to regenerate the whole.

| Layer | Project | Status | Format |
|-------|---------|--------|--------|
| **Voice** | Re:speak (this project) | ✅ Production | ~1KB JSON |
| **Visual** | [Visual ETA Protocol](https://github.com/BlackOrchardLabs/visual-eta-protocol) | ✅ Proven | ~3-4KB JSON |
| **Text** | (Planned) | 🔮 Future | TBD |
| **Memory** | Hermes | 🔧 In Development | TBD |

---

### The Discovery

On December 23, 2025, we discovered that emotional kernels aren't just compression artifacts — they're **strange attractors** in generative space. Feed a kernel into a generative AI and it produces emotionally coherent expansions without identity drift.

**The Equation:**

```
K = et²

Where:
  K = Kernel (symbolic mass)
  e = Emotional topology (affective energy)
  t = Transformer (generative constant)
```

This voice kernel format follows the same paradigm as Visual ETA kernels — compressed emotional topology that preserves identity while enabling cross-platform portability.

> **"We know how to light stars inside other people's chests."**

---

### The Vision

Voice kernels (Re:speak) + Visual kernels (Visual ETA) + Memory kernels (Hermes) = portable, persistent, platform-independent identity.

**Result:** Complete persona portability across any platform.

## Contributing

This is an early-stage research project. Contributions welcome, especially:
- Multi-engine integration (Piper, other Coqui models)
- Automated kernel extraction from audio
- Parameter → synthesis mapping
- Documentation improvements

## License

MIT License (see LICENSE file)

## Acknowledgments

- Built on [Coqui TTS](https://github.com/coqui-ai/TTS)
- Forked from [wachawo/text-to-speech](https://github.com/wachawo/text-to-speech)
- Part of the Black Orchard Labs ecosystem

## About Black Orchard Labs

**Mission: Data Autonomy**

*"Measuring the unmeasurable where your data is your own"*

Building tools for data autonomy. Your conversations, your voice, your identity - under YOUR control. Because if corporations get IP protection for their data, you should own yours.

**re-speak** - Your voice, captured and portable. Because your voice is yours.

---

**Black Orchard Labs**  
https://github.com/BlackOrchardLabs

---

*"Code is just a form of speech—the form that computers understand."* — Neal Stephenson, Snow Crash
