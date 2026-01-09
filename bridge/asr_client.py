"""
Nemotron ASR WebSocket Client
=============================

Handles real-time streaming from Nemotron Speech ASR.
Target latency: 80ms with att_context_size=[70,0]
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Callable
from collections import deque

try:
    import websockets
    import numpy as np
except ImportError:
    raise ImportError("Install: pip install websockets numpy")


@dataclass
class TranscriptChunk:
    """A single transcript chunk from ASR."""
    text: str
    is_final: bool
    confidence: float
    timestamp_ms: float
    latency_ms: float
    word_timings: list = field(default_factory=list)

    @classmethod
    def from_nemotron(cls, message: bytes | str, recv_time_ms: float) -> "TranscriptChunk":
        """Parse Nemotron ASR response."""
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')

            data = json.loads(message)

            return cls(
                text=data.get("text", ""),
                is_final=data.get("is_final", False),
                confidence=data.get("confidence", 1.0),
                timestamp_ms=recv_time_ms,
                latency_ms=data.get("latency_ms", 0),
                word_timings=data.get("word_timings", [])
            )
        except json.JSONDecodeError:
            # Plain text response
            return cls(
                text=str(message),
                is_final=True,
                confidence=1.0,
                timestamp_ms=recv_time_ms,
                latency_ms=0
            )


class NemotronASRClient:
    """
    WebSocket client for Nemotron Speech ASR.

    Usage:
        client = NemotronASRClient()
        async for chunk in client.stream(audio_source):
            print(chunk.text)
    """

    def __init__(
        self,
        url: str = "ws://localhost:8080",
        sample_rate: int = 16000,
        chunk_ms: int = 80
    ):
        self.url = url
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = int(sample_rate * chunk_ms / 1000)

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._on_transcript: Optional[Callable[[TranscriptChunk], None]] = None

        # Stats
        self.latencies = deque(maxlen=100)

    async def connect(self) -> bool:
        """Establish connection to ASR service."""
        try:
            self._ws = await websockets.connect(
                self.url,
                open_timeout=10,
                ping_interval=20,
                ping_timeout=10
            )
            self._running = True
            return True
        except Exception as e:
            print(f"ASR connection failed: {e}")
            return False

    async def disconnect(self):
        """Close connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to ASR."""
        if not self._ws:
            raise RuntimeError("Not connected")
        await self._ws.send(audio_bytes)

    async def receive(self) -> Optional[TranscriptChunk]:
        """Receive single transcript chunk."""
        if not self._ws:
            return None

        try:
            message = await asyncio.wait_for(self._ws.recv(), timeout=0.5)
            recv_time = time.perf_counter() * 1000
            chunk = TranscriptChunk.from_nemotron(message, recv_time)

            if chunk.latency_ms:
                self.latencies.append(chunk.latency_ms)

            return chunk

        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed:
            self._running = False
            return None

    async def stream(
        self,
        audio_source: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        """
        Stream audio to ASR and yield transcripts.

        Args:
            audio_source: Async iterator yielding audio bytes
                         (16kHz, mono, int16)

        Yields:
            TranscriptChunk for each ASR response
        """
        if not await self.connect():
            raise RuntimeError("Failed to connect to ASR")

        try:
            # Start receive task
            recv_queue: asyncio.Queue[TranscriptChunk] = asyncio.Queue()

            async def receiver():
                while self._running:
                    chunk = await self.receive()
                    if chunk and chunk.text.strip():
                        await recv_queue.put(chunk)

            recv_task = asyncio.create_task(receiver())

            # Send audio and yield transcripts
            async for audio_bytes in audio_source:
                await self.send_audio(audio_bytes)

                # Check for transcripts (non-blocking)
                while not recv_queue.empty():
                    yield await recv_queue.get()

            # Drain remaining transcripts
            recv_task.cancel()
            while not recv_queue.empty():
                yield await recv_queue.get()

        finally:
            await self.disconnect()

    async def stream_microphone(self) -> AsyncIterator[TranscriptChunk]:
        """
        Stream from default microphone.

        Yields:
            TranscriptChunk for each ASR response
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("Install: pip install sounddevice")

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_data = (indata[:, 0] * 32767).astype(np.int16)
            asyncio.get_event_loop().call_soon_threadsafe(
                audio_queue.put_nowait,
                audio_data.tobytes()
            )

        async def audio_source():
            while self._running:
                try:
                    yield await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=audio_callback
        )

        with stream:
            async for chunk in self.stream(audio_source()):
                yield chunk

    @property
    def avg_latency_ms(self) -> float:
        """Average latency over recent chunks."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running
