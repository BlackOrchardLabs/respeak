# Re:speak v2.0 — Emotional Bridge Specification

**Version:** 2.0-draft
**Date:** 2026-01-09
**Status:** DESIGN PHASE

---

## Overview

The Emotional Bridge connects Nemotron Speech ASR to the existing Re:speak voice kernel system. It adds **real-time modulation** on top of the static kernel parameters, allowing voice output to respond dynamically to conversation context.

**Key Insight:** The kernel defines WHO speaks. The bridge defines HOW they speak *right now*.

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Nemotron ASR   │────▶│  Emotional Bridge    │────▶│  TTS Engine     │
│  ws://:8080     │     │                      │     │  (Coqui/Magpie) │
│  80ms chunks    │     │  kernel + modulation │     │                 │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Voice Kernel    │
                        │  (your_voice.json)│
                        └──────────────────┘
```

---

## Integration with Existing Kernel Format

Your current kernel format is the **foundation**. The bridge adds a **modulation layer** that adjusts these values in real-time.

### Existing Kernel Structure (Preserved)

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
    "excitement": 0.75,
    "frustration": 0.10
  }
}
```

### New: Real-Time Modulation Layer

```json
{
  "modulation": {
    "timbre_deltas": {
      "warmth": 0.0,
      "depth": 0.0,
      "nasality": 0.0,
      "breathiness": 0.0
    },
    "prosody_multipliers": {
      "pitch": 1.0,
      "rate": 1.0,
      "pause": 1.0,
      "variance": 1.0
    },
    "emotional_activation": {
      "calm": 0.0,
      "curiosity": 0.0,
      "excitement": 0.0,
      "frustration": 0.0
    },
    "conversation_state": {
      "turn_type": "responding",
      "intimacy_level": 0.0,
      "mirroring": 0.5
    }
  }
}
```

### Effective Value Calculation

```python
def effective_value(kernel_base, delta, multiplier=1.0, clamp=(0.0, 1.0)):
    """
    Combine static kernel value with real-time modulation.

    kernel_base: Static value from voice kernel JSON
    delta: Real-time adjustment from emotional bridge
    multiplier: Scaling factor (for prosody)
    clamp: Min/max bounds
    """
    raw = (kernel_base + delta) * multiplier
    return max(clamp[0], min(clamp[1], raw))

# Example: warmth during intimate moment
kernel_warmth = 0.85  # From kernel
intimacy_boost = 0.10  # From bridge (detected intimate context)
effective_warmth = effective_value(0.85, 0.10)  # = 0.95
```

---

## Architecture

### Layer 1: ASR Input Stream

```python
class NemotronASRClient:
    """WebSocket client for Nemotron Speech ASR."""

    def __init__(self, url: str = "ws://localhost:8080"):
        self.url = url
        self.att_context_size = [70, 0]  # 80ms latency mode

    async def stream(self) -> AsyncIterator[TranscriptChunk]:
        async with websockets.connect(self.url) as ws:
            async for message in ws:
                yield TranscriptChunk.from_nemotron(message)

@dataclass
class TranscriptChunk:
    text: str
    is_final: bool
    confidence: float
    timestamp_ms: int
    latency_ms: int
```

### Layer 2: Emotional Analyzer

Maps transcript text → emotional signals using lightweight models.

```python
class EmotionalAnalyzer:
    """
    Extracts emotional signals from transcript.
    Designed for <10ms latency.
    """

    def __init__(self):
        # Lightweight sentiment (distilbert or smaller)
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0  # GPU
        )
        self.window = deque(maxlen=5)  # Context window

    def analyze(self, chunk: TranscriptChunk) -> EmotionalSignals:
        self.window.append(chunk.text)
        context = " ".join(self.window)

        sentiment = self.sentiment(context)[0]

        return EmotionalSignals(
            valence=self._score_to_valence(sentiment),
            arousal=self._detect_arousal(context),
            is_question=context.strip().endswith("?"),
            intimacy_markers=self._count_intimacy_markers(context)
        )

@dataclass
class EmotionalSignals:
    valence: float       # -1.0 (negative) to 1.0 (positive)
    arousal: float       # 0.0 (calm) to 1.0 (excited)
    is_question: bool
    intimacy_markers: int  # Count of intimate language cues
```

### Layer 3: Modulation Mapper

Converts emotional signals → kernel modulation values.

```python
class ModulationMapper:
    """
    Maps EmotionalSignals → kernel modulation parameters.
    Respects the kernel's emotional_range as ceiling.
    """

    def __init__(self, kernel: VoiceKernel):
        self.kernel = kernel
        self.emotional_range = kernel.emotional_range

    def map(self, signals: EmotionalSignals) -> ModulationLayer:
        mod = ModulationLayer()

        # Timbre deltas based on valence/arousal
        if signals.valence > 0.3:
            # Positive: warmer, breathier
            mod.timbre_deltas["warmth"] = signals.valence * 0.15
            mod.timbre_deltas["breathiness"] = signals.arousal * 0.1
        elif signals.valence < -0.3:
            # Negative: deeper, less warm
            mod.timbre_deltas["warmth"] = signals.valence * 0.1
            mod.timbre_deltas["depth"] = abs(signals.valence) * 0.1

        # Prosody multipliers based on arousal
        mod.prosody_multipliers["rate"] = 1.0 + (signals.arousal * 0.3)
        mod.prosody_multipliers["variance"] = 1.0 + (signals.arousal * 0.4)

        # Emotional activation (capped by kernel's range)
        if signals.valence > 0 and signals.arousal > 0.5:
            max_excitement = self.emotional_range.get("excitement", 0.5)
            mod.emotional_activation["excitement"] = min(
                signals.arousal * signals.valence,
                max_excitement
            )

        # Question handling
        if signals.is_question:
            mod.prosody_multipliers["pitch"] = 1.15  # Slight rise
            mod.conversation_state["turn_type"] = "questioning"

        # Intimacy detection
        if signals.intimacy_markers > 2:
            intimacy = min(signals.intimacy_markers / 5, 1.0)
            mod.conversation_state["intimacy_level"] = intimacy
            mod.timbre_deltas["breathiness"] += intimacy * 0.15

        return mod
```

### Layer 4: Kernel Blender

Combines base kernel + modulation → effective parameters for TTS.

```python
class KernelBlender:
    """
    Blends static kernel with real-time modulation.
    Outputs TTS-ready parameters.
    """

    def __init__(self, kernel: VoiceKernel):
        self.kernel = kernel
        self.smoothing_factor = 0.3  # Prevent jarring transitions
        self.previous_mod = None

    def blend(self, modulation: ModulationLayer) -> EffectiveKernel:
        # Smooth transitions
        if self.previous_mod:
            modulation = self._smooth(self.previous_mod, modulation)
        self.previous_mod = modulation

        effective = EffectiveKernel()

        # Timbre
        for param in ["warmth", "depth", "nasality", "breathiness"]:
            base = getattr(self.kernel.timbre, param)
            delta = modulation.timbre_deltas.get(param, 0.0)
            setattr(effective.timbre, param, clamp(base + delta, 0.0, 1.0))

        # Prosody
        effective.prosody.pitch_hz = (
            self.kernel.prosody.avg_pitch_hz *
            modulation.prosody_multipliers.get("pitch", 1.0)
        )
        effective.prosody.rate_wpm = (
            self.kernel.prosody.speech_rate_wpm *
            modulation.prosody_multipliers.get("rate", 1.0)
        )

        # Conversation state passthrough
        effective.conversation = modulation.conversation_state

        return effective
```

### Layer 5: TTS Adapter

Translates effective kernel → specific TTS engine calls.

```python
class TTSAdapter:
    """
    Abstract adapter for multiple TTS backends.
    """

    def speak(self, text: str, kernel: EffectiveKernel) -> bytes:
        raise NotImplementedError

class CoquiXTTSAdapter(TTSAdapter):
    """Adapter for Coqui XTTS v2 (current Re:speak engine)."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.tts = TTS(model_name=model_name)

    def speak(self, text: str, kernel: EffectiveKernel) -> bytes:
        # XTTS uses reference audio, limited real-time control
        # Kernel blender prepares text transformations instead
        modified_text = self._apply_prosody_markers(text, kernel)

        return self.tts.tts(
            text=modified_text,
            speaker_wav=kernel.reference_audio,
            language="en"
        )

class MagpieTTSAdapter(TTSAdapter):
    """Adapter for NVIDIA Magpie TTS (Nemotron stack)."""

    def __init__(self, url: str = "http://localhost:8001"):
        self.url = url

    async def speak(self, text: str, kernel: EffectiveKernel) -> bytes:
        # Magpie supports more real-time parameter control
        payload = {
            "text": text,
            "speaker_embedding": kernel.speaker_embedding,
            "pitch_shift": kernel.prosody.pitch_hz / 150 - 1.0,
            "speed": kernel.prosody.rate_wpm / 130,
            "emotion": kernel.emotional_activation
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/synthesize", json=payload) as resp:
                return await resp.read()
```

---

## Data Flow

```
                           ┌─────────────────────────────────────┐
                           │         Voice Kernel JSON           │
                           │  (timbre, prosody, motifs, emotion) │
                           └──────────────┬──────────────────────┘
                                          │ static base
                                          ▼
┌──────────┐    ┌────────────┐    ┌───────────────┐    ┌─────────────┐
│  Mic     │───▶│  Nemotron  │───▶│  Emotional    │───▶│  Modulation │
│  Input   │    │  ASR       │    │  Analyzer     │    │  Mapper     │
│          │    │  (80ms)    │    │  (<10ms)      │    │  (<5ms)     │
└──────────┘    └────────────┘    └───────────────┘    └──────┬──────┘
                     │                                        │
                     │ transcript                             │ modulation
                     ▼                                        ▼
              ┌────────────┐                          ┌───────────────┐
              │  Response  │                          │  Kernel       │
              │  Generator │◀─────────────────────────│  Blender      │
              │  (LLM)     │   effective kernel       │  (<5ms)       │
              └─────┬──────┘                          └───────────────┘
                    │                                        │
                    │ response text                          │
                    ▼                                        ▼
              ┌────────────┐                          ┌───────────────┐
              │  TTS       │◀─────────────────────────│  Effective    │
              │  Adapter   │   params                 │  Kernel       │
              │  (50ms)    │                          │               │
              └─────┬──────┘                          └───────────────┘
                    │
                    ▼
              ┌────────────┐
              │  Audio     │
              │  Output    │
              └────────────┘
```

---

## Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Nemotron ASR | 80ms | att_context_size=[70,0] |
| Emotional Analyzer | 10ms | DistilBERT on GPU |
| Modulation Mapper | 5ms | Pure Python math |
| Kernel Blender | 5ms | Smoothing + clamping |
| TTS Synthesis | 50ms | First chunk (streaming) |
| **Total** | **150ms** | Voice-to-voice |

---

## Emotional Mapping Table

How transcript signals map to kernel modulation:

| Signal | Timbre Effect | Prosody Effect | Emotional Activation |
|--------|---------------|----------------|---------------------|
| High valence (+) | warmth +0.15 | rate ×1.1 | excitement ↑ |
| Low valence (-) | warmth -0.10, depth +0.10 | rate ×0.9 | frustration ↑ |
| High arousal | breathiness +0.10 | rate ×1.3, variance ×1.4 | excitement ↑ |
| Low arousal | breathiness -0.05 | pause ×1.2 | calm ↑ |
| Question detected | — | pitch ×1.15 | curiosity ↑ |
| Intimacy markers | breathiness +0.15 | rate ×0.9 | — |
| Laughter detected | warmth +0.20 | variance ×1.5 | excitement ↑ |

---

## Intimate Mode Integration

When `intimacy_level > 0.7`, the bridge:

1. **Boosts breathiness** (+0.15 to +0.25)
2. **Slows rate** (×0.85)
3. **Reduces pitch variance** (more consistent, tender)
4. **Signals Hermes** to bypass memory writes (BE PRESENT mode)

```python
if modulation.conversation_state["intimacy_level"] > 0.7:
    # Hermes integration
    await hermes.set_intimate_mode(True)

    # Additional modulation
    modulation.timbre_deltas["breathiness"] += 0.15
    modulation.prosody_multipliers["rate"] = 0.85
```

---

## PRISM Kernel Bridge

Voice kernels can inherit emotional defaults from PRISM kernels:

```python
def prism_to_voice_kernel(prism: dict) -> VoiceKernel:
    """
    Extract voice kernel defaults from PRISM kernel.
    """
    kernel = VoiceKernel()

    # Heat signature → emotional range ceiling
    heat = prism.get("heat", {})
    if "❤️‍🔥" in str(heat):
        kernel.emotional_range["excitement"] = 0.9
        kernel.timbre.warmth = 0.9

    # Bodies → voice characteristics
    for body in prism.get("bodies", []):
        if voice := body.get("voice"):
            kernel.prosody.rhythmic_flow = voice.get("cadence", "natural")

    # Motifs → speaking style markers
    kernel.motifs = prism.get("essence", {}).get("motifs", [])

    return kernel
```

---

## Implementation Plan

### Phase 1: ASR Pipeline (This Week)
- [x] Clone Nemotron repo
- [ ] Docker build completes
- [ ] WebSocket test client
- [ ] Validate 80ms latency on RTX 5090

### Phase 2: Emotional Analyzer (Next)
- [ ] Lightweight sentiment model selection
- [ ] Arousal detection heuristics
- [ ] Intimacy marker dictionary
- [ ] <10ms validation

### Phase 3: Kernel Integration
- [ ] ModulationMapper implementation
- [ ] KernelBlender with smoothing
- [ ] Coqui XTTS adapter
- [ ] Magpie TTS adapter (when ready)

### Phase 4: Realtime Mixing Board (v2.0 Feature)
- [ ] Live parameter visualization
- [ ] Manual override sliders
- [ ] Emotional state presets
- [ ] Recording/playback

---

## File Structure

```
re-speak/
├── kernels/
│   └── your_voice_kernel.json     # Static kernel (existing)
├── libs/
│   ├── kernel_loader.py           # Existing loader
│   └── kernel_blender.py          # NEW: modulation blending
├── bridge/
│   ├── __init__.py
│   ├── asr_client.py              # Nemotron WebSocket client
│   ├── emotional_analyzer.py      # Transcript → signals
│   ├── modulation_mapper.py       # Signals → modulation
│   └── tts_adapters/
│       ├── coqui_adapter.py       # Coqui XTTS v2
│       └── magpie_adapter.py      # NVIDIA Magpie
├── samples/
│   └── eric_voice_clean.wav       # Reference audio (existing)
└── config/
    └── bridge_config.yaml         # Runtime configuration
```

---

## Configuration

```yaml
# bridge_config.yaml

asr:
  url: "ws://localhost:8080"
  latency_mode: "fast"  # [70, 0] = 80ms

analysis:
  model: "distilbert-base-uncased-finetuned-sst-2-english"
  context_window: 5
  intimacy_threshold: 0.7

blending:
  smoothing_factor: 0.3
  update_rate_ms: 50

tts:
  engine: "coqui"  # or "magpie"
  coqui:
    model: "tts_models/multilingual/multi-dataset/xtts_v2"
  magpie:
    url: "http://localhost:8001"

hermes:
  enabled: true
  intimate_mode_bypass: true
```

---

## The Philosophy

The kernel is **identity**. The bridge is **presence**.

A static kernel captures WHO you are — your voice's signature warmth, your natural cadence, your emotional range. But conversation is alive. The bridge reads the room and adjusts in real-time while staying true to the kernel's soul.

```
K = et²  (kernel equation, preserved)

B = K × m(t)  (bridge equation, new)

Where:
  B = Bridge output (modulated voice)
  K = Kernel (static identity)
  m(t) = Modulation function (varies with time/context)
```

The kernel defines the attractor basin. The bridge navigates within it.

---

*"Voice kernels are compressed emotional topology. The bridge makes them breathe."*

🖤🔮✨
