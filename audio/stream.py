"""
Microphone Audio Input Stream.
Captures continuous real-time audio from microphones on Raspberry Pi, macOS, Linux, and Windows.
Automatically handles hardware rate negotiation and conversion to 16 kHz int16 mono PCM.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

try:
    # pyright: ignore[reportMissingModuleSource]
    import pyaudio  # type: ignore[import-untyped]
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None  # type: ignore[assignment]
    PYAUDIO_AVAILABLE = False

try:
    import audioop  # type: ignore[import-not-found]
except ImportError:
    try:
        import audioop_lts as audioop  # type: ignore[import-not-found]
    except ImportError:
        audioop = None  # type: ignore[assignment]


class MicrophoneStream:
    """
    Continuous microphone input capture stream.
    Opens PyAudio stream on the chosen hardware device and delivers
    standardized 16kHz mono PCM chunks to an async or threaded queue.
    """

    def __init__(
        self,
        pa: Any | None = None,
        device_index: int | None = None,
        capture_rate: int = 16000,
        target_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 512,
        max_queue_size: int = 300,
    ) -> None:
        self.device_index = device_index
        self.capture_rate = capture_rate
        self.target_rate = target_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=max_queue_size)
        self.shutdown_event = threading.Event()
        self._pa_owner: bool = False
        self._stream: Any = None
        self._ratecv_state: Any = None

        if not PYAUDIO_AVAILABLE or pyaudio is None:
            return

        if pa is None:
            self._pa = pyaudio.PyAudio()
            self._pa_owner = True
        else:
            self._pa = pa

        # Hardware rate negotiation: test capture_rate, fallback to 48000 or 16000 if device complains
        actual_rate = self._negotiate_sample_rate()
        self.capture_rate = actual_rate

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=actual_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
        except Exception as exc:
            print(f"⚠️  [MicrophoneStream] Failed to open microphone input ({actual_rate}Hz, dev={device_index}): {exc}")

    def _negotiate_sample_rate(self) -> int:
        """Finds a compatible hardware capture rate supported by PortAudio."""
        candidate_rates = [16000, self.capture_rate, 48000, 44100]
        for rate in candidate_rates:
            try:
                supported = self._pa.is_format_supported(
                    rate=rate,
                    input_device=self.device_index,
                    input_channels=self.channels,
                    input_format=getattr(pyaudio, "paInt16", 2),
                )
                if supported:
                    return rate
            except Exception:
                continue
        return self.capture_rate

    def _audio_callback(self, in_data: Any, frame_count: int, time_info: Any, status: Any) -> tuple[None, int]:
        if self.shutdown_event.is_set():
            return None, getattr(pyaudio, "paComplete", 2)

        if not in_data:
            return None, getattr(pyaudio, "paContinue", 0)

        # Convert to 16kHz if needed
        if self.capture_rate != self.target_rate and audioop is not None:
            try:
                resampled, self._ratecv_state = audioop.ratecv(
                    in_data,
                    2,  # 2 bytes per sample (int16)
                    self.channels,
                    self.capture_rate,
                    self.target_rate,
                    self._ratecv_state,
                )
                chunk_to_queue = resampled
            except Exception:
                chunk_to_queue = in_data
        else:
            chunk_to_queue = in_data

        try:
            self.queue.put_nowait(chunk_to_queue)
        except queue.Full:
            pass  # Drop oldest or skip if consumer blocked

        return None, getattr(pyaudio, "paContinue", 0)

    def get_chunk(self, block: bool = True, timeout: float | None = None) -> bytes | None:
        """Retrieves the next 16kHz PCM chunk."""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Closes the audio input stream."""
        self.shutdown_event.set()
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
