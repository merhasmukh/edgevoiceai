"""
Voice Activity Detection (VAD) & Energy Analysis.
Provides RMS calculation, noise gating, rolling syllable onset pre-buffering,
and interruption streak detection for full-duplex conversational voice agents.
"""

from __future__ import annotations

import collections
import math
import struct
from dataclasses import dataclass


def compute_rms(pcm_bytes: bytes) -> float:
    """
    Computes Root Mean Square (RMS) volume of 16-bit signed PCM audio bytes.
    Returns 0.0 for empty or malformed byte sequences.
    """
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    try:
        samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
        return math.sqrt(sum(s * s for s in samples) / n)
    except Exception:
        return 0.0


@dataclass(slots=True)
class VADConfig:
    """Configuration parameters for Voice Activity Detection and barge-in."""
    interrupt_rms_threshold: int = 3000
    ai_interrupt_threshold: int = 5000
    noise_gate_threshold: int = 300
    hang_frames: int = 25
    streak_trigger: int = 3
    pre_buffer_max_frames: int = 8


class VoiceActivityDetector:
    """
    Stateful VAD processor that tracks speech streaks, gate status,
    and preserves syllable onset frames across conversational turns.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.hang_frames_remaining: int = 0
        self.gate_open: bool = self.config.noise_gate_threshold <= 0
        self.pre_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self.config.pre_buffer_max_frames
        )
        self.interruption_streak: int = 0

    def reset(self) -> None:
        """Resets dynamic tracking state."""
        self.hang_frames_remaining = 0
        self.gate_open = self.config.noise_gate_threshold <= 0
        self.pre_buffer.clear()
        self.interruption_streak = 0

    def process_chunk(self, pcm_chunk: bytes) -> tuple[bool, list[bytes]]:
        """
        Processes a raw incoming PCM chunk against the noise gate.

        Returns:
            (should_stream, frames_to_emit)
            - should_stream: True if gate is currently open.
            - frames_to_emit: Pre-buffered onset frames + current chunk when
              gate transitions from closed to open.
        """
        if self.config.noise_gate_threshold <= 0:
            return True, [pcm_chunk]

        rms = compute_rms(pcm_chunk)
        frames_to_emit: list[bytes] = []

        if rms >= self.config.noise_gate_threshold:
            self.hang_frames_remaining = self.config.hang_frames
            if not self.gate_open:
                # Gate just opened: flush onset frames so words like "Yes" / "No" aren't clipped
                self.gate_open = True
                while self.pre_buffer:
                    frames_to_emit.append(self.pre_buffer.popleft())
            frames_to_emit.append(pcm_chunk)
            return True, frames_to_emit

        # Below threshold: check hang frames
        if self.hang_frames_remaining > 0:
            self.hang_frames_remaining -= 1
            return True, [pcm_chunk]

        # Gate is closed
        self.gate_open = False
        self.pre_buffer.append(pcm_chunk)
        return False, []

    def check_barge_in(self, pcm_chunk: bytes, threshold: int | None = None) -> bool:
        """
        Checks whether incoming audio volume continuously exceeds the interruption
        threshold for the configured streak trigger count.
        """
        thresh = threshold if threshold is not None else self.config.interrupt_rms_threshold
        if thresh <= 0:
            return False

        rms = compute_rms(pcm_chunk)
        if rms >= thresh:
            self.interruption_streak += 1
            if self.interruption_streak >= self.config.streak_trigger:
                self.interruption_streak = 0
                return True
        else:
            self.interruption_streak = max(0, self.interruption_streak - 1)

        return False
