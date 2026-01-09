#!/usr/bin/env python3
"""
Nemotron ASR Latency Test Client
================================

Tests real-time transcription latency on RTX 5090.
Target: sub-100ms voice-to-transcript.

Usage:
    python test_asr_latency.py              # Interactive mic test
    python test_asr_latency.py --file audio.wav  # Test with file
    python test_asr_latency.py --benchmark  # Run latency benchmark

Requirements:
    pip install websockets sounddevice numpy
"""

import asyncio
import argparse
import json
import time
import sys
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque
import statistics

try:
    import websockets
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install websockets sounddevice numpy")
    sys.exit(1)


# Configuration
ASR_URL = "ws://localhost:8080"
SAMPLE_RATE = 16000  # Nemotron expects 16kHz
CHUNK_DURATION_MS = 80  # Match att_context_size=[70,0]
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


@dataclass
class LatencyStats:
    """Tracks latency measurements."""
    samples: List[float] = field(default_factory=list)

    def add(self, latency_ms: float):
        self.samples.append(latency_ms)

    def report(self) -> dict:
        if not self.samples:
            return {"error": "No samples collected"}

        return {
            "count": len(self.samples),
            "min_ms": round(min(self.samples), 2),
            "max_ms": round(max(self.samples), 2),
            "mean_ms": round(statistics.mean(self.samples), 2),
            "median_ms": round(statistics.median(self.samples), 2),
            "p95_ms": round(sorted(self.samples)[int(len(self.samples) * 0.95)], 2) if len(self.samples) >= 20 else None,
            "target_met": statistics.mean(self.samples) < 100
        }


class ASRLatencyTester:
    """Tests Nemotron ASR WebSocket latency."""

    def __init__(self, url: str = ASR_URL):
        self.url = url
        self.stats = LatencyStats()
        self.running = False
        self.transcript_buffer = deque(maxlen=50)

    async def test_connection(self) -> bool:
        """Verify ASR service is available."""
        try:
            async with websockets.connect(self.url, open_timeout=5) as ws:
                # Send a tiny audio chunk to verify
                test_audio = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
                await ws.send(test_audio)

                # Wait for response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    print(f"✓ ASR service responding at {self.url}")
                    return True
                except asyncio.TimeoutError:
                    print(f"✓ ASR service connected (no speech detected)")
                    return True

        except Exception as e:
            print(f"✗ Cannot connect to ASR: {e}")
            print(f"  Make sure Nemotron container is running:")
            print(f"  ./scripts/nemotron.sh start")
            return False

    async def stream_microphone(self):
        """Stream live microphone audio to ASR."""
        print("\n🎤 Starting microphone stream...")
        print("   Speak into your microphone. Press Ctrl+C to stop.\n")

        self.running = True
        audio_queue = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            # Convert to int16 and queue
            audio_data = (indata[:, 0] * 32767).astype(np.int16)
            asyncio.get_event_loop().call_soon_threadsafe(
                audio_queue.put_nowait, audio_data.tobytes()
            )

        try:
            async with websockets.connect(self.url) as ws:
                # Start audio stream
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype=np.float32,
                    blocksize=CHUNK_SIZE,
                    callback=audio_callback
                )

                with stream:
                    # Send/receive tasks
                    send_task = asyncio.create_task(
                        self._send_audio(ws, audio_queue)
                    )
                    recv_task = asyncio.create_task(
                        self._receive_transcripts(ws)
                    )

                    await asyncio.gather(send_task, recv_task)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.running = False

    async def _send_audio(self, ws, audio_queue: asyncio.Queue):
        """Send audio chunks with timestamps."""
        while self.running:
            try:
                audio_data = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.5
                )

                # Record send time
                send_time = time.perf_counter() * 1000

                # Send audio with timestamp metadata
                await ws.send(audio_data)

                # Store timestamp for latency calculation
                # (In production, use a proper correlation mechanism)

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break

    async def _receive_transcripts(self, ws):
        """Receive and display transcripts with latency."""
        while self.running:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                recv_time = time.perf_counter() * 1000

                # Parse response
                try:
                    data = json.loads(response)
                    transcript = data.get("text", "")
                    is_final = data.get("is_final", False)
                    latency = data.get("latency_ms", 0)

                    if latency:
                        self.stats.add(latency)

                    # Display
                    marker = "█" if is_final else "░"
                    latency_str = f"[{latency:.0f}ms]" if latency else ""

                    if transcript.strip():
                        print(f"{marker} {transcript} {latency_str}")
                        self.transcript_buffer.append(transcript)

                except json.JSONDecodeError:
                    # Raw text response
                    if response.strip():
                        print(f"░ {response}")

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break

    async def benchmark(self, duration_seconds: int = 10):
        """Run automated latency benchmark with synthetic audio."""
        print(f"\n📊 Running {duration_seconds}s latency benchmark...")
        print(f"   Sending {CHUNK_DURATION_MS}ms chunks at {SAMPLE_RATE}Hz\n")

        self.stats = LatencyStats()
        chunks_sent = 0

        try:
            async with websockets.connect(self.url) as ws:
                start_time = time.perf_counter()

                while (time.perf_counter() - start_time) < duration_seconds:
                    # Generate test audio (white noise with speech-like envelope)
                    chunk = self._generate_test_audio()

                    send_time = time.perf_counter() * 1000
                    await ws.send(chunk)
                    chunks_sent += 1

                    # Receive response
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        recv_time = time.perf_counter() * 1000

                        latency = recv_time - send_time
                        self.stats.add(latency)

                        # Progress indicator
                        if chunks_sent % 10 == 0:
                            print(f"   Chunk {chunks_sent}: {latency:.1f}ms")

                    except asyncio.TimeoutError:
                        pass

                    # Maintain real-time pace
                    await asyncio.sleep(CHUNK_DURATION_MS / 1000)

        except Exception as e:
            print(f"Benchmark error: {e}")

        # Report
        print("\n" + "="*50)
        print("LATENCY BENCHMARK RESULTS")
        print("="*50)

        report = self.stats.report()
        for key, value in report.items():
            print(f"  {key}: {value}")

        print("="*50)

        if report.get("target_met"):
            print("✓ TARGET MET: Mean latency < 100ms")
        else:
            print("✗ TARGET MISSED: Mean latency >= 100ms")

        return report

    def _generate_test_audio(self) -> bytes:
        """Generate speech-like test audio."""
        # Mix of frequencies that trigger ASR
        t = np.linspace(0, CHUNK_DURATION_MS / 1000, CHUNK_SIZE)

        # Fundamental + harmonics (speech-like)
        audio = (
            0.3 * np.sin(2 * np.pi * 150 * t) +  # F0
            0.2 * np.sin(2 * np.pi * 300 * t) +  # H1
            0.1 * np.sin(2 * np.pi * 450 * t) +  # H2
            0.05 * np.random.randn(CHUNK_SIZE)   # Noise
        )

        # Normalize and convert
        audio = (audio / np.max(np.abs(audio)) * 0.8 * 32767).astype(np.int16)
        return audio.tobytes()

    async def test_file(self, filepath: str):
        """Test ASR with audio file."""
        import wave

        print(f"\n📁 Testing with file: {filepath}")

        try:
            with wave.open(filepath, 'rb') as wf:
                # Verify format
                if wf.getframerate() != SAMPLE_RATE:
                    print(f"⚠ Warning: File is {wf.getframerate()}Hz, expected {SAMPLE_RATE}Hz")

                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

                # Convert stereo to mono if needed
                if wf.getnchannels() == 2:
                    audio_data = audio_data[::2]

        except Exception as e:
            print(f"Error reading file: {e}")
            return

        print(f"   Duration: {len(audio_data) / SAMPLE_RATE:.2f}s")
        print(f"   Chunks: {len(audio_data) // CHUNK_SIZE}\n")

        self.stats = LatencyStats()

        try:
            async with websockets.connect(self.url) as ws:
                # Send chunks
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    chunk = audio_data[i:i + CHUNK_SIZE]
                    if len(chunk) < CHUNK_SIZE:
                        # Pad final chunk
                        chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

                    send_time = time.perf_counter() * 1000
                    await ws.send(chunk.tobytes())

                    # Receive
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        recv_time = time.perf_counter() * 1000

                        latency = recv_time - send_time
                        self.stats.add(latency)

                        # Parse and display
                        try:
                            data = json.loads(response)
                            text = data.get("text", "")
                            if text.strip():
                                print(f"   [{latency:.0f}ms] {text}")
                        except json.JSONDecodeError:
                            if response.strip():
                                print(f"   [{latency:.0f}ms] {response}")

                    except asyncio.TimeoutError:
                        pass

                    # Real-time pace
                    await asyncio.sleep(CHUNK_DURATION_MS / 1000)

        except Exception as e:
            print(f"Error: {e}")

        # Report
        print("\n" + "-"*40)
        report = self.stats.report()
        print(f"Results: {report}")


async def main():
    parser = argparse.ArgumentParser(description="Test Nemotron ASR latency")
    parser.add_argument("--url", default=ASR_URL, help="ASR WebSocket URL")
    parser.add_argument("--file", help="Test with audio file instead of mic")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration (seconds)")

    args = parser.parse_args()

    tester = ASRLatencyTester(url=args.url)

    # Test connection first
    print("="*50)
    print("NEMOTRON ASR LATENCY TESTER")
    print(f"Target: <100ms (att_context_size=[70,0])")
    print("="*50)

    if not await tester.test_connection():
        sys.exit(1)

    # Run selected test
    if args.benchmark:
        await tester.benchmark(duration_seconds=args.duration)
    elif args.file:
        await tester.test_file(args.file)
    else:
        await tester.stream_microphone()

        # Show final stats
        print("\n" + "="*50)
        print("SESSION STATS")
        print("="*50)
        report = tester.stats.report()
        for key, value in report.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
