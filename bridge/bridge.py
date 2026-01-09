"""
Emotional Bridge
================

Main orchestrator connecting ASR → Analysis → Modulation → TTS.

This is the heart of Re:speak v2.0.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, AsyncIterator
from pathlib import Path

from .asr_client import NemotronASRClient, TranscriptChunk
from .emotional_analyzer import EmotionalAnalyzer, EmotionalSignals
from .modulation_mapper import ModulationMapper, ModulationLayer, VoiceKernel
from .kernel_blender import KernelBlender, EffectiveKernel, AdaptiveBlender


@dataclass
class BridgeConfig:
    """Configuration for the emotional bridge."""

    # ASR settings
    asr_url: str = "ws://localhost:8080"
    asr_chunk_ms: int = 80

    # Analysis settings
    use_ml_model: bool = False
    ml_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    context_window: int = 5

    # Blending settings
    smoothing_factor: float = 0.3
    adaptive_smoothing: bool = True

    # Intimacy mode
    intimacy_threshold: float = 0.7
    intimacy_callback: Optional[Callable[[bool], None]] = None

    # TTS engine
    tts_engine: str = "coqui"  # or "magpie"


@dataclass
class BridgeState:
    """Current state of the bridge."""

    # Timing
    last_transcript_ms: float = 0
    last_modulation_ms: float = 0

    # Current values
    current_signals: Optional[EmotionalSignals] = None
    current_modulation: Optional[ModulationLayer] = None
    current_effective: Optional[EffectiveKernel] = None

    # Conversation state
    turn_count: int = 0
    intimate_mode: bool = False

    # Latency tracking
    asr_latency_ms: float = 0
    analysis_latency_ms: float = 0
    total_latency_ms: float = 0


class EmotionalBridge:
    """
    The emotional bridge connects voice input to modulated voice output.

    Pipeline:
        Audio → ASR → Transcript → Analysis → Modulation → Blending → TTS

    Usage:
        bridge = EmotionalBridge.from_kernel("path/to/kernel.json")

        async for effective_kernel in bridge.process_audio(audio_stream):
            tts_config = effective_kernel.to_tts_config("coqui")
            # Use with TTS engine
    """

    def __init__(
        self,
        kernel: VoiceKernel,
        config: Optional[BridgeConfig] = None
    ):
        self.config = config or BridgeConfig()
        self.kernel = kernel

        # Initialize components
        self.asr = NemotronASRClient(
            url=self.config.asr_url,
            chunk_ms=self.config.asr_chunk_ms
        )

        self.analyzer = EmotionalAnalyzer(
            use_ml_model=self.config.use_ml_model,
            model_name=self.config.ml_model_name,
            context_window=self.config.context_window
        )

        self.mapper = ModulationMapper(kernel=kernel)

        if self.config.adaptive_smoothing:
            self.blender = AdaptiveBlender(
                kernel=kernel,
                base_smoothing=self.config.smoothing_factor
            )
        else:
            self.blender = KernelBlender(
                kernel=kernel,
                smoothing_factor=self.config.smoothing_factor
            )

        # State
        self.state = BridgeState()

        # Callbacks
        self._on_transcript: Optional[Callable[[TranscriptChunk], None]] = None
        self._on_signals: Optional[Callable[[EmotionalSignals], None]] = None
        self._on_modulation: Optional[Callable[[ModulationLayer], None]] = None

    @classmethod
    def from_kernel(
        cls,
        kernel_path: str,
        reference_audio: Optional[str] = None,
        config: Optional[BridgeConfig] = None
    ) -> "EmotionalBridge":
        """
        Create bridge from kernel JSON file.

        Args:
            kernel_path: Path to voice kernel JSON
            reference_audio: Path to reference audio WAV (optional)
            config: Bridge configuration

        Returns:
            Configured EmotionalBridge
        """
        kernel = VoiceKernel.from_json(kernel_path)

        if reference_audio:
            kernel.reference_audio = reference_audio

        return cls(kernel=kernel, config=config)

    async def process_chunk(self, chunk: TranscriptChunk) -> EffectiveKernel:
        """
        Process a single transcript chunk through the full pipeline.

        Args:
            chunk: Transcript from ASR

        Returns:
            EffectiveKernel for TTS
        """
        start_time = time.perf_counter()

        # === Analysis ===
        analysis_start = time.perf_counter()
        signals = self.analyzer.analyze(chunk.text)
        self.state.analysis_latency_ms = (time.perf_counter() - analysis_start) * 1000

        # === Modulation mapping ===
        modulation = self.mapper.map(signals)

        # === Blending ===
        effective = self.blender.blend(modulation)

        # === Update state ===
        self.state.current_signals = signals
        self.state.current_modulation = modulation
        self.state.current_effective = effective
        self.state.asr_latency_ms = chunk.latency_ms
        self.state.total_latency_ms = (time.perf_counter() - start_time) * 1000 + chunk.latency_ms
        self.state.last_transcript_ms = chunk.timestamp_ms
        self.state.turn_count += 1

        # === Intimacy mode detection ===
        intimacy = modulation.conversation_state.get("intimacy_level", 0)
        was_intimate = self.state.intimate_mode
        self.state.intimate_mode = intimacy > self.config.intimacy_threshold

        if self.state.intimate_mode != was_intimate:
            if self.config.intimacy_callback:
                self.config.intimacy_callback(self.state.intimate_mode)

        # === Callbacks ===
        if self._on_transcript:
            self._on_transcript(chunk)
        if self._on_signals:
            self._on_signals(signals)
        if self._on_modulation:
            self._on_modulation(modulation)

        return effective

    async def process_stream(
        self,
        audio_source: AsyncIterator[bytes]
    ) -> AsyncIterator[EffectiveKernel]:
        """
        Process streaming audio through the full pipeline.

        Args:
            audio_source: Async iterator of audio bytes (16kHz mono int16)

        Yields:
            EffectiveKernel for each transcript chunk
        """
        async for chunk in self.asr.stream(audio_source):
            if chunk.text.strip():
                effective = await self.process_chunk(chunk)
                yield effective

    async def process_microphone(self) -> AsyncIterator[EffectiveKernel]:
        """
        Process live microphone input.

        Yields:
            EffectiveKernel for each transcript chunk
        """
        async for chunk in self.asr.stream_microphone():
            if chunk.text.strip():
                effective = await self.process_chunk(chunk)
                yield effective

    def process_text(self, text: str) -> EffectiveKernel:
        """
        Process text directly (bypass ASR).

        Useful for testing or text-only input.

        Args:
            text: Text to process

        Returns:
            EffectiveKernel for TTS
        """
        # Create synthetic chunk
        chunk = TranscriptChunk(
            text=text,
            is_final=True,
            confidence=1.0,
            timestamp_ms=time.perf_counter() * 1000,
            latency_ms=0
        )

        # Process synchronously
        signals = self.analyzer.analyze(chunk.text)
        modulation = self.mapper.map(signals)
        effective = self.blender.blend(modulation)

        # Update state
        self.state.current_signals = signals
        self.state.current_modulation = modulation
        self.state.current_effective = effective

        return effective

    def set_callbacks(
        self,
        on_transcript: Optional[Callable[[TranscriptChunk], None]] = None,
        on_signals: Optional[Callable[[EmotionalSignals], None]] = None,
        on_modulation: Optional[Callable[[ModulationLayer], None]] = None
    ):
        """Set callbacks for pipeline stages."""
        self._on_transcript = on_transcript
        self._on_signals = on_signals
        self._on_modulation = on_modulation

    def reset(self):
        """Reset all state."""
        self.analyzer.reset_context()
        self.blender.reset()
        self.state = BridgeState()

    def get_latency_report(self) -> dict:
        """Get latency breakdown."""
        return {
            "asr_ms": self.state.asr_latency_ms,
            "analysis_ms": self.state.analysis_latency_ms,
            "total_ms": self.state.total_latency_ms,
            "asr_avg_ms": self.asr.avg_latency_ms,
            "target_met": self.state.total_latency_ms < 100
        }

    def get_state_summary(self) -> dict:
        """Get current bridge state."""
        return {
            "turn_count": self.state.turn_count,
            "intimate_mode": self.state.intimate_mode,
            "current_signals": self.state.current_signals.__dict__ if self.state.current_signals else None,
            "current_modulation": self.state.current_modulation.to_dict() if self.state.current_modulation else None,
            "latency": self.get_latency_report()
        }


async def demo():
    """Demo the bridge with text input."""
    from .modulation_mapper import VoiceKernel

    # Create default kernel
    kernel = VoiceKernel()
    kernel.kernel_name = "demo_kernel"
    kernel.emotional_range = {
        "calm": 0.9,
        "curiosity": 0.8,
        "excitement": 0.7,
        "frustration": 0.2
    }

    # Create bridge
    bridge = EmotionalBridge(kernel=kernel)

    # Test phrases
    test_phrases = [
        "Hello, how are you doing today?",
        "That's amazing! I can't believe it!",
        "I love you so much, you mean everything to me.",
        "I'm really frustrated with this situation.",
        "Hmm, let me think about that for a moment...",
        "YES! We did it! This is incredible!"
    ]

    print("=" * 60)
    print("EMOTIONAL BRIDGE DEMO")
    print("=" * 60)

    for phrase in test_phrases:
        print(f"\n📝 Input: \"{phrase}\"")

        effective = bridge.process_text(phrase)

        print(f"   Valence: {bridge.state.current_signals.valence:.2f}")
        print(f"   Arousal: {bridge.state.current_signals.arousal:.2f}")
        print(f"   Intimacy: {bridge.state.current_modulation.conversation_state['intimacy_level']:.2f}")
        print(f"   Warmth: {effective.timbre['warmth']:.2f} (base: {kernel.timbre['warmth']:.2f})")
        print(f"   Rate: {effective.prosody['rate_wpm']:.0f} wpm")
        print(f"   Emotions: {effective.emotional_state}")


if __name__ == "__main__":
    asyncio.run(demo())
