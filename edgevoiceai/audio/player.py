"""
Thread-safe PCM Audio Output Player.
Streams int16 PCM audio to output speakers via non-blocking PortAudio callbacks.
Supports instant clear() for barge-in interruptions and graceful channel/rate padding.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

try:
    # pyright: ignore[reportMissingModuleSource]
    import pyaudio  # type: ignore[import-untyped]
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None  # type: ignore[assignment]
    PYAUDIO_AVAILABLE = False


class AudioPlayer:
    """
    Thread-safe speaker audio output player.
    Accepts raw PCM chunks from asyncio worker tasks and serves them to
    PortAudio's real-time callback thread.
    """

    def __init__(
        self,
        pa: Any | None = None,
        device_index: int | None = None,
        sample_rate: int = 24000,
        channels: int = 1,
        frames_per_buffer: int = 512,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._buf: collections.deque[bytes] = collections.deque()
        self._lock = threading.Lock()
        self._last_play_time: float = 0.0
        self._closed: bool = False
        self._stream: Any = None
        self._pa_owner: bool = False

        if not PYAUDIO_AVAILABLE or pyaudio is None:
            return

        if pa is None:
            self._pa = pyaudio.PyAudio()
            self._pa_owner = True
        else:
            self._pa = pa

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                output=True,
                output_device_index=device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._callback,
            )
            self._stream.start_stream()
        except Exception as exc:
            print(f"⚠️  [AudioPlayer] Failed to open audio output stream ({sample_rate}Hz, dev={device_index}): {exc}")

    def _callback(self, in_data: Any, frame_count: int, time_info: Any, status: Any) -> tuple[bytes, int]:
        needed = frame_count * 2 * self.channels  # int16 = 2 bytes per sample
        out = bytearray()
        with self._lock:
            while len(out) < needed and self._buf:
                chunk = self._buf[0]
                take = needed - len(out)
                out += chunk[:take]
                if take >= len(chunk):
                    self._buf.popleft()
                else:
                    self._buf[0] = chunk[take:]
            if out:
                self._last_play_time = time.time()

        # Silence pad if buffer starved
        if len(out) < needed:
            out += b"\x00" * (needed - len(out))

        return bytes(out), getattr(pyaudio, "paContinue", 0)

    def feed(self, pcm_bytes: bytes) -> None:
        """Pushes raw PCM audio into the playback queue."""
        if not pcm_bytes or self._closed:
            return
        with self._lock:
            self._buf.append(pcm_bytes)
            self._last_play_time = time.time()

    def is_playing(self) -> bool:
        """Returns True if audio is actively streaming or draining from the buffer."""
        with self._lock:
            if bool(self._buf):
                return True
            return (time.time() - self._last_play_time) < 0.2

    def clear(self) -> None:
        """Instantly clears all queued audio (called upon user barge-in)."""
        with self._lock:
            self._buf.clear()
            self._last_play_time = 0.0

    def close(self) -> None:
        """Shuts down the PortAudio output stream."""
        self._closed = True
        self.clear()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa_owner and self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
