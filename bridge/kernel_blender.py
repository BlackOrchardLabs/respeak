"""
Kernel Blender
==============

Combines static voice kernel with real-time modulation.
Outputs TTS-ready parameters with smooth transitions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import copy


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


@dataclass
class EffectiveKernel:
    """
    Blended kernel ready for TTS.

    Combines static kernel base with real-time modulation.
    """

    # Effective timbre (after modulation)
    timbre: Dict[str, float] = field(default_factory=lambda: {
        "warmth": 0.5,
        "depth": 0.5,
        "nasality": 0.1,
        "breathiness": 0.2
    })

    # Effective prosody (after modulation)
    prosody: Dict[str, float] = field(default_factory=lambda: {
        "pitch_hz": 150,
        "rate_wpm": 130,
        "pause_density": 0.25,
        "variance": 0.5
    })

    # Active emotional state
    emotional_state: Dict[str, float] = field(default_factory=dict)

    # Conversation context
    conversation: Dict[str, Any] = field(default_factory=lambda: {
        "turn_type": "responding",
        "intimacy_level": 0.0,
        "mirroring": 0.5
    })

    # Reference audio for TTS
    reference_audio: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timbre": self.timbre,
            "prosody": self.prosody,
            "emotional_state": self.emotional_state,
            "conversation": self.conversation,
            "reference_audio": self.reference_audio
        }

    def to_tts_config(self, engine: str = "coqui") -> dict:
        """
        Convert to TTS engine-specific config.

        Args:
            engine: "coqui" or "magpie"

        Returns:
            Engine-specific configuration dict
        """
        if engine == "coqui":
            return self._to_coqui_config()
        elif engine == "magpie":
            return self._to_magpie_config()
        else:
            return self.to_dict()

    def _to_coqui_config(self) -> dict:
        """Coqui XTTS config (limited real-time control)."""
        # XTTS has limited runtime parameters
        # Most modulation happens via text preprocessing
        return {
            "speaker_wav": self.reference_audio,
            "language": "en",
            # Speed can be adjusted post-synthesis
            "speed_factor": self.prosody["rate_wpm"] / 130
        }

    def _to_magpie_config(self) -> dict:
        """Magpie TTS config (full parameter control)."""
        return {
            "speaker_embedding": None,  # Set by adapter
            "pitch_shift": (self.prosody["pitch_hz"] / 150) - 1.0,
            "speed": self.prosody["rate_wpm"] / 130,
            "energy": self.timbre["warmth"],
            "emotion_vector": self.emotional_state
        }


class KernelBlender:
    """
    Blends static kernel with real-time modulation.

    Features:
    - Smooth parameter transitions
    - Modulation clamping
    - State interpolation
    """

    def __init__(
        self,
        kernel,  # VoiceKernel
        smoothing_factor: float = 0.3,
        update_rate_ms: int = 50
    ):
        self.kernel = kernel
        self.smoothing_factor = smoothing_factor
        self.update_rate_ms = update_rate_ms

        self._previous_effective: Optional[EffectiveKernel] = None

    def blend(self, modulation) -> EffectiveKernel:
        """
        Blend kernel base with modulation layer.

        Args:
            modulation: ModulationLayer with adjustments

        Returns:
            EffectiveKernel ready for TTS
        """
        effective = EffectiveKernel()

        # === Timbre blending ===
        for param in ["warmth", "depth", "nasality", "breathiness"]:
            base = self.kernel.timbre.get(param, 0.5)
            delta = modulation.timbre_deltas.get(param, 0.0)
            effective.timbre[param] = clamp(base + delta)

        # === Prosody blending ===
        effective.prosody["pitch_hz"] = (
            self.kernel.prosody.get("avg_pitch_hz", 150) *
            modulation.prosody_multipliers.get("pitch", 1.0)
        )
        effective.prosody["rate_wpm"] = (
            self.kernel.prosody.get("speech_rate_wpm", 130) *
            modulation.prosody_multipliers.get("rate", 1.0)
        )
        effective.prosody["pause_density"] = (
            self.kernel.prosody.get("pause_density", 0.25) *
            modulation.prosody_multipliers.get("pause", 1.0)
        )
        effective.prosody["variance"] = (
            self.kernel.prosody.get("pitch_variance", 0.5) *
            modulation.prosody_multipliers.get("variance", 1.0)
        )

        # === Emotional state ===
        effective.emotional_state = copy.deepcopy(modulation.emotional_activation)

        # === Conversation state ===
        effective.conversation = copy.deepcopy(modulation.conversation_state)

        # === Reference audio ===
        effective.reference_audio = self.kernel.reference_audio

        # === Smoothing ===
        if self._previous_effective:
            effective = self._smooth(self._previous_effective, effective)

        self._previous_effective = effective
        return effective

    def _smooth(
        self,
        previous: EffectiveKernel,
        current: EffectiveKernel
    ) -> EffectiveKernel:
        """
        Smooth transition between parameter states.

        Uses exponential moving average for natural transitions.
        """
        alpha = self.smoothing_factor  # 0.3 = 30% new, 70% old

        smoothed = EffectiveKernel()

        # Smooth timbre
        for param in previous.timbre:
            prev_val = previous.timbre.get(param, 0.5)
            curr_val = current.timbre.get(param, 0.5)
            smoothed.timbre[param] = prev_val * (1 - alpha) + curr_val * alpha

        # Smooth prosody
        for param in previous.prosody:
            prev_val = previous.prosody.get(param, 0.0)
            curr_val = current.prosody.get(param, 0.0)
            smoothed.prosody[param] = prev_val * (1 - alpha) + curr_val * alpha

        # Smooth emotional state
        all_emotions = set(previous.emotional_state.keys()) | set(current.emotional_state.keys())
        for emotion in all_emotions:
            prev_val = previous.emotional_state.get(emotion, 0.0)
            curr_val = current.emotional_state.get(emotion, 0.0)
            smoothed.emotional_state[emotion] = prev_val * (1 - alpha) + curr_val * alpha

        # Conversation state (no smoothing, instant update)
        smoothed.conversation = current.conversation

        # Reference audio
        smoothed.reference_audio = current.reference_audio

        return smoothed

    def reset(self):
        """Reset smoothing state."""
        self._previous_effective = None

    def set_kernel(self, kernel):
        """Update the base kernel."""
        self.kernel = kernel
        self.reset()

    def set_smoothing(self, factor: float):
        """Adjust smoothing factor (0.0-1.0)."""
        self.smoothing_factor = clamp(factor)


class AdaptiveBlender(KernelBlender):
    """
    Blender with adaptive smoothing based on change magnitude.

    Large changes = faster transition (more responsive)
    Small changes = slower transition (more stable)
    """

    def __init__(self, kernel, base_smoothing: float = 0.3):
        super().__init__(kernel, smoothing_factor=base_smoothing)
        self.base_smoothing = base_smoothing
        self.min_smoothing = 0.1
        self.max_smoothing = 0.5

    def _smooth(
        self,
        previous: EffectiveKernel,
        current: EffectiveKernel
    ) -> EffectiveKernel:
        """Smooth with adaptive factor based on change magnitude."""

        # Calculate change magnitude
        total_change = 0.0
        count = 0

        for param in previous.timbre:
            prev_val = previous.timbre.get(param, 0.5)
            curr_val = current.timbre.get(param, 0.5)
            total_change += abs(curr_val - prev_val)
            count += 1

        avg_change = total_change / max(count, 1)

        # Larger change = faster response (higher alpha)
        adaptive_alpha = self.base_smoothing + (avg_change * 0.5)
        adaptive_alpha = clamp(adaptive_alpha, self.min_smoothing, self.max_smoothing)

        # Store for this blend
        original_smoothing = self.smoothing_factor
        self.smoothing_factor = adaptive_alpha

        result = super()._smooth(previous, current)

        self.smoothing_factor = original_smoothing
        return result
