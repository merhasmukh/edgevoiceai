"""
Configuration classes for Voice AI Framework.
Defines parameters for audio streaming, model setup, VAD, and edge devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from edgevoiceai.notifications.whatsapp import WhatsAppConfig


@dataclass(slots=True)
class VoiceConfig:
    """
    Central configuration object for a VoiceAgent instance.
    Compatible with Raspberry Pi, Linux, macOS, and Windows.
    """
    api_key: str = ""
    model: str = "gemini-3.1-flash-live-preview"
    voice: str = "Puck"
    system_instruction: str = (
        "You are a real-time conversational Voice AI assistant. "
        "Speak naturally, warmly, and concisely (1 to 2 spoken sentences per turn). "
        "Respond directly like in a real-time phone call, avoiding bullet points or long monologues."
    )
    greeting_message: str | None = "Hello! I am ready to assist you."

    # Audio Hardware Device Routing (independent input/output)
    input_device: int | str | None = None
    output_device: int | str | None = None

    # Sample Rates
    mic_capture_rate: int = 16000
    gemini_input_rate: int = 16000
    speaker_sample_rate: int = 24000

    # VAD & Barge-in Interruption Thresholds (RMS)
    interrupt_rms_threshold: int = 2500
    ai_interrupt_threshold: int = 2800
    noise_gate_threshold: int = 250
    hang_frames: int = 15
    vad_streak_trigger: int = 4
    pre_buffer_max_frames: int = 8

    # Network & Resilience
    reconnect_delay_seconds: float = 3.0
    ws_base_url: str | None = None
    open_timeout: float = 30.0
    close_timeout: float = 5.0
    ping_interval: float | None = None

    # Storage & Media
    media_directory: str = "audio_experiences"

    # Optional WhatsApp Notification Configuration
    whatsapp: WhatsAppConfig | None = None

    # Optional metadata or custom user parameters
    metadata: dict[str, Any] = field(default_factory=dict)

