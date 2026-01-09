"""
Modulation Mapper
=================

Maps EmotionalSignals → kernel modulation parameters.
Respects kernel's emotional_range as ceiling.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class ModulationLayer:
    """Real-time modulation on top of static kernel."""

    # Timbre adjustments (additive deltas)
    timbre_deltas: Dict[str, float] = field(default_factory=lambda: {
        "warmth": 0.0,
        "depth": 0.0,
        "nasality": 0.0,
        "breathiness": 0.0
    })

    # Prosody scaling (multiplicative)
    prosody_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "pitch": 1.0,
        "rate": 1.0,
        "pause": 1.0,
        "variance": 1.0
    })

    # Emotional activation levels
    emotional_activation: Dict[str, float] = field(default_factory=lambda: {
        "calm": 0.0,
        "curiosity": 0.0,
        "excitement": 0.0,
        "frustration": 0.0,
        "tenderness": 0.0
    })

    # Conversation state
    conversation_state: Dict[str, Any] = field(default_factory=lambda: {
        "turn_type": "responding",
        "intimacy_level": 0.0,
        "mirroring": 0.5
    })

    def to_dict(self) -> dict:
        return {
            "timbre_deltas": self.timbre_deltas,
            "prosody_multipliers": self.prosody_multipliers,
            "emotional_activation": self.emotional_activation,
            "conversation_state": self.conversation_state
        }


@dataclass
class VoiceKernel:
    """Voice kernel from Re:speak JSON format."""

    kernel_name: str = "default"
    version: str = "1.0"

    timbre: Dict[str, float] = field(default_factory=lambda: {
        "warmth": 0.5,
        "depth": 0.5,
        "nasality": 0.1,
        "breathiness": 0.2
    })

    prosody: Dict[str, Any] = field(default_factory=lambda: {
        "avg_pitch_hz": 150,
        "pitch_variance": 0.5,
        "speech_rate_wpm": 130,
        "pause_density": 0.25,
        "rhythmic_flow": "natural"
    })

    motifs: list = field(default_factory=list)

    emotional_range: Dict[str, float] = field(default_factory=lambda: {
        "calm": 0.8,
        "curiosity": 0.7,
        "excitement": 0.6,
        "frustration": 0.3
    })

    reference_audio: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "VoiceKernel":
        """Load kernel from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        kernel = cls()
        kernel.kernel_name = data.get("kernel_name", "unnamed")
        kernel.version = data.get("version", "1.0")
        kernel.timbre = data.get("timbre", kernel.timbre)
        kernel.prosody = data.get("prosody", kernel.prosody)
        kernel.motifs = data.get("motifs", [])
        kernel.emotional_range = data.get("emotional_range", kernel.emotional_range)

        return kernel


class ModulationMapper:
    """
    Maps EmotionalSignals → ModulationLayer.

    Respects the kernel's emotional_range as ceiling.
    """

    def __init__(self, kernel: Optional[VoiceKernel] = None):
        self.kernel = kernel or VoiceKernel()

    def map(self, signals) -> ModulationLayer:
        """
        Convert emotional signals to modulation parameters.

        Args:
            signals: EmotionalSignals from analyzer

        Returns:
            ModulationLayer with computed adjustments
        """
        mod = ModulationLayer()

        # === Timbre from valence ===
        if signals.valence > 0.3:
            # Positive: warmer, slightly breathier
            mod.timbre_deltas["warmth"] = signals.valence * 0.15
            mod.timbre_deltas["breathiness"] = signals.arousal * 0.08
        elif signals.valence < -0.3:
            # Negative: less warm, deeper
            mod.timbre_deltas["warmth"] = signals.valence * 0.10
            mod.timbre_deltas["depth"] = abs(signals.valence) * 0.08

        # === Prosody from arousal ===
        # Higher arousal = faster, more varied
        mod.prosody_multipliers["rate"] = 1.0 + (signals.arousal - 0.5) * 0.4
        mod.prosody_multipliers["variance"] = 1.0 + (signals.arousal - 0.5) * 0.5

        # Lower arousal = more pauses
        if signals.arousal < 0.3:
            mod.prosody_multipliers["pause"] = 1.3

        # === Question handling ===
        if signals.is_question:
            mod.prosody_multipliers["pitch"] = 1.12  # Rising intonation
            mod.conversation_state["turn_type"] = "questioning"
            mod.emotional_activation["curiosity"] = min(
                0.7,
                self.kernel.emotional_range.get("curiosity", 0.7)
            )

        # === Exclamation handling ===
        if signals.is_exclamation:
            mod.prosody_multipliers["variance"] = 1.4
            mod.emotional_activation["excitement"] = min(
                signals.arousal,
                self.kernel.emotional_range.get("excitement", 0.6)
            )

        # === Hesitation ===
        if signals.hesitation_count > 0:
            mod.prosody_multipliers["rate"] = max(0.85, mod.prosody_multipliers["rate"] - 0.1)
            mod.prosody_multipliers["pause"] = 1.2

        # === Intimacy ===
        if signals.intimacy_level > 0.5:
            mod.conversation_state["intimacy_level"] = signals.intimacy_level
            mod.timbre_deltas["breathiness"] += signals.intimacy_level * 0.12
            mod.prosody_multipliers["rate"] *= 0.9  # Slower, softer
            mod.emotional_activation["tenderness"] = signals.intimacy_level

        # === Emotional activation (capped by kernel range) ===
        # Map valence + arousal to specific emotions
        if signals.valence > 0 and signals.arousal > 0.5:
            max_excitement = self.kernel.emotional_range.get("excitement", 0.5)
            mod.emotional_activation["excitement"] = min(
                signals.valence * signals.arousal,
                max_excitement
            )
        elif signals.valence > 0 and signals.arousal < 0.4:
            max_calm = self.kernel.emotional_range.get("calm", 0.8)
            mod.emotional_activation["calm"] = min(
                signals.valence * (1 - signals.arousal),
                max_calm
            )
        elif signals.valence < -0.3:
            max_frustration = self.kernel.emotional_range.get("frustration", 0.3)
            mod.emotional_activation["frustration"] = min(
                abs(signals.valence) * signals.arousal,
                max_frustration
            )

        return mod

    def update_kernel(self, kernel: VoiceKernel):
        """Update the reference kernel."""
        self.kernel = kernel


# Precomputed modulation presets
MODULATION_PRESETS = {
    "neutral": ModulationLayer(),

    "warm": ModulationLayer(
        timbre_deltas={"warmth": 0.15, "breathiness": 0.05},
        prosody_multipliers={"rate": 0.95},
        emotional_activation={"calm": 0.6}
    ),

    "excited": ModulationLayer(
        timbre_deltas={"warmth": 0.10},
        prosody_multipliers={"rate": 1.2, "variance": 1.4, "pitch": 1.08},
        emotional_activation={"excitement": 0.8}
    ),

    "intimate": ModulationLayer(
        timbre_deltas={"warmth": 0.20, "breathiness": 0.15},
        prosody_multipliers={"rate": 0.85, "variance": 0.8},
        emotional_activation={"tenderness": 0.9},
        conversation_state={"intimacy_level": 0.9}
    ),

    "concerned": ModulationLayer(
        timbre_deltas={"warmth": 0.10, "depth": 0.05},
        prosody_multipliers={"rate": 0.9, "pause": 1.3},
        emotional_activation={"calm": 0.4}
    ),

    "curious": ModulationLayer(
        prosody_multipliers={"pitch": 1.1, "variance": 1.2},
        emotional_activation={"curiosity": 0.7},
        conversation_state={"turn_type": "questioning"}
    )
}
