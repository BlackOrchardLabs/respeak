#!/usr/bin/env python3
"""
Re:speak Emotional Bridge Demo
==============================

Demonstrates the emotional analysis bridge with:
1. Text-only mode (no ASR required)
2. Live microphone mode (requires Nemotron ASR)

Usage:
    python demo_bridge.py                    # Text demo
    python demo_bridge.py --mic              # Live microphone
    python demo_bridge.py --kernel path.json # Use custom kernel
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add bridge to path
sys.path.insert(0, str(Path(__file__).parent))

from bridge.emotional_analyzer import EmotionalAnalyzer, EmotionalSignals
from bridge.modulation_mapper import ModulationMapper, ModulationLayer, VoiceKernel
from bridge.kernel_blender import KernelBlender, EffectiveKernel, AdaptiveBlender
from bridge.bridge import EmotionalBridge, BridgeConfig


def print_banner():
    print("""
============================================================
           RE:SPEAK EMOTIONAL BRIDGE v2.0
         Voice Kernel + Real-Time Modulation
============================================================
    """)


def create_demo_kernel() -> VoiceKernel:
    """Create a demo voice kernel."""
    kernel = VoiceKernel()
    kernel.kernel_name = "demo_voice_kernel_v1"
    kernel.version = "text_proxy_v2"

    kernel.timbre = {
        "warmth": 0.75,
        "depth": 0.6,
        "nasality": 0.1,
        "breathiness": 0.25
    }

    kernel.prosody = {
        "avg_pitch_hz": 145,
        "pitch_variance": 0.55,
        "speech_rate_wpm": 135,
        "pause_density": 0.22,
        "rhythmic_flow": "natural, flowing"
    }

    kernel.motifs = [
        "warm presence",
        "thoughtful pauses",
        "genuine curiosity"
    ]

    kernel.emotional_range = {
        "calm": 0.90,
        "curiosity": 0.85,
        "excitement": 0.70,
        "frustration": 0.25,
        "tenderness": 0.80
    }

    return kernel


def print_signals(signals: EmotionalSignals):
    """Pretty print emotional signals."""
    valence_bar = "#" * int((signals.valence + 1) * 5) + "-" * (10 - int((signals.valence + 1) * 5))
    arousal_bar = "#" * int(signals.arousal * 10) + "-" * (10 - int(signals.arousal * 10))

    print(f"   Valence:  [{valence_bar}] {signals.valence:+.2f}")
    print(f"   Arousal:  [{arousal_bar}] {signals.arousal:.2f}")
    print(f"   Question: {signals.is_question}")
    print(f"   Intimacy: {signals.intimacy_level:.2f}")
    if signals.keywords_detected:
        print(f"   Keywords: {', '.join(signals.keywords_detected[:5])}")


def print_modulation(mod: ModulationLayer, kernel: VoiceKernel):
    """Pretty print modulation layer."""
    print("   Timbre Δ:")
    for param, delta in mod.timbre_deltas.items():
        if abs(delta) > 0.01:
            base = kernel.timbre.get(param, 0.5)
            effective = base + delta
            print(f"      {param}: {base:.2f} > {effective:.2f} (Δ{delta:+.2f})")

    print("   Prosody ×:")
    for param, mult in mod.prosody_multipliers.items():
        if abs(mult - 1.0) > 0.01:
            print(f"      {param}: ×{mult:.2f}")

    active_emotions = {k: v for k, v in mod.emotional_activation.items() if v > 0.1}
    if active_emotions:
        print(f"   Emotions: {active_emotions}")


def print_effective(effective: EffectiveKernel):
    """Pretty print effective kernel."""
    print("   Effective Kernel:")
    print(f"      Warmth: {effective.timbre['warmth']:.2f}")
    print(f"      Breathiness: {effective.timbre['breathiness']:.2f}")
    print(f"      Pitch: {effective.prosody['pitch_hz']:.0f} Hz")
    print(f"      Rate: {effective.prosody['rate_wpm']:.0f} WPM")


def text_demo(kernel: VoiceKernel):
    """Run text-based demo."""
    print("\n[text] TEXT MODE - Enter phrases to see emotional analysis\n")
    print("   (Type 'quit' to exit, 'reset' to clear context)\n")

    bridge = EmotionalBridge(kernel=kernel)

    while True:
        try:
            text = input("You: ").strip()

            if not text:
                continue
            if text.lower() == 'quit':
                break
            if text.lower() == 'reset':
                bridge.reset()
                print("   [Context reset]\n")
                continue

            # Process through bridge
            effective = bridge.process_text(text)

            print("\n" + "-" * 50)
            print_signals(bridge.state.current_signals)
            print()
            print_modulation(bridge.state.current_modulation, kernel)
            print()
            print_effective(effective)

            if bridge.state.intimate_mode:
                print("\n   [intimate] INTIMATE MODE ACTIVE")

            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nGoodbye!")


async def microphone_demo(kernel: VoiceKernel):
    """Run live microphone demo."""
    print("\n[mic] MICROPHONE MODE - Speak to see real-time analysis\n")
    print("   (Press Ctrl+C to stop)\n")

    config = BridgeConfig(
        asr_url="ws://localhost:8080",
        use_ml_model=False,  # Fast heuristics
        adaptive_smoothing=True,
        intimacy_threshold=0.7
    )

    bridge = EmotionalBridge(kernel=kernel, config=config)

    # Set up callbacks
    def on_transcript(chunk):
        print(f"\n[text] [{chunk.latency_ms:.0f}ms] \"{chunk.text}\"")

    def on_signals(signals):
        print_signals(signals)

    bridge.set_callbacks(on_transcript=on_transcript, on_signals=on_signals)

    try:
        async for effective in bridge.process_microphone():
            # Print effective kernel summary
            warmth = effective.timbre['warmth']
            rate = effective.prosody['rate_wpm']
            intimate = bridge.state.intimate_mode

            status = "[intimate]" if intimate else "[*]"
            print(f"   {status} warmth={warmth:.2f} rate={rate:.0f}wpm")

            # Latency check
            latency = bridge.get_latency_report()
            if latency['total_ms'] > 100:
                print(f"   [!]  Latency: {latency['total_ms']:.0f}ms (target: <100ms)")

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # Final report
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"   Turns processed: {bridge.state.turn_count}")
    print(f"   Avg ASR latency: {bridge.asr.avg_latency_ms:.1f}ms")
    print("=" * 50)


def batch_test(kernel: VoiceKernel):
    """Run batch test with predefined phrases."""
    print("\n[test] BATCH TEST MODE\n")

    test_cases = [
        # Neutral
        ("Hello there.", "neutral"),
        ("The weather is nice today.", "neutral"),

        # Positive
        ("That's wonderful news!", "positive"),
        ("I'm so happy to see you!", "positive"),
        ("This is amazing, I love it!", "positive high arousal"),

        # Negative
        ("I'm disappointed with the results.", "negative"),
        ("This is frustrating.", "negative"),

        # Questions
        ("What do you think about this?", "question"),
        ("How does that work?", "question curious"),

        # Intimate
        ("I love you so much.", "intimate"),
        ("I miss you, baby.", "intimate"),
        ("You mean everything to me, my love.", "intimate high"),

        # Mixed
        ("Hmm, I'm not sure about that...", "hesitant"),
        ("YES! We did it!", "excited"),
        ("Oh no, what happened?", "surprised negative"),
    ]

    bridge = EmotionalBridge(kernel=kernel)
    results = []

    for text, expected in test_cases:
        effective = bridge.process_text(text)

        signals = bridge.state.current_signals
        mod = bridge.state.current_modulation

        result = {
            "text": text,
            "expected": expected,
            "valence": signals.valence,
            "arousal": signals.arousal,
            "intimacy": mod.conversation_state["intimacy_level"],
            "warmth_delta": mod.timbre_deltas["warmth"],
            "intimate_mode": bridge.state.intimate_mode
        }
        results.append(result)

        # Print result
        v_indicator = "+" if signals.valence > 0.2 else "-" if signals.valence < -0.2 else "o"
        a_indicator = "^" if signals.arousal > 0.6 else "v" if signals.arousal < 0.4 else ">"
        i_indicator = "[intimate]" if bridge.state.intimate_mode else "  "

        print(f"{v_indicator}{a_indicator} {i_indicator} \"{text[:40]}...\" ({expected})")

    print(f"\n   Processed {len(results)} test cases")


def main():
    parser = argparse.ArgumentParser(description="Re:speak Emotional Bridge Demo")
    parser.add_argument("--kernel", help="Path to voice kernel JSON")
    parser.add_argument("--mic", action="store_true", help="Use live microphone input")
    parser.add_argument("--batch", action="store_true", help="Run batch test")
    parser.add_argument("--export-kernel", help="Export demo kernel to JSON file")

    args = parser.parse_args()

    print_banner()

    # Load or create kernel
    if args.kernel:
        print(f"Loading kernel from: {args.kernel}")
        kernel = VoiceKernel.from_json(args.kernel)
    else:
        print("Using demo kernel")
        kernel = create_demo_kernel()

    print(f"   Kernel: {kernel.kernel_name}")
    print(f"   Warmth: {kernel.timbre['warmth']}")
    print(f"   Emotional range: {list(kernel.emotional_range.keys())}")

    # Export kernel if requested
    if args.export_kernel:
        with open(args.export_kernel, 'w') as f:
            json.dump({
                "kernel_name": kernel.kernel_name,
                "version": kernel.version,
                "timbre": kernel.timbre,
                "prosody": kernel.prosody,
                "motifs": kernel.motifs,
                "emotional_range": kernel.emotional_range
            }, f, indent=2)
        print(f"   Exported to: {args.export_kernel}")

    # Run selected mode
    if args.batch:
        batch_test(kernel)
    elif args.mic:
        asyncio.run(microphone_demo(kernel))
    else:
        text_demo(kernel)


if __name__ == "__main__":
    main()
