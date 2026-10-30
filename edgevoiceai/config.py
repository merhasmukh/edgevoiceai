"""
Configuration classes for Voice AI Framework.
Defines parameters for audio streaming, model setup, VAD, edge devices, notifications, and RAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from edgevoiceai.notifications.whatsapp import WhatsAppConfig


@dataclass
class AudioConfig:
    """
    Configuration for microphone input, speaker output, and Voice Activity Detection (VAD).
    """
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

    # Storage & Local Audio Media
    media_directory: str = "audio_experiences"


@dataclass
class RAGConfig:
    """
    Configuration for the RAG (Retrieval-Augmented Generation) subsystem and vector stores.
    """
    backend: str = "memory"  # "memory", "qdrant", "pinecone"
    embedding_model: str = "gemini-embedding-001"
    dimension: int = 768
    collection_name: str = "knowledge_base"
    default_limit: int = 3
    default_score_threshold: float = 0.45
    qdrant_url: str | None = None
    pinecone_index: str | None = None
    pinecone_namespace: str = ""


@dataclass
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

    # Modular sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    rag: RAGConfig | None = None
    whatsapp: WhatsAppConfig | None = None

    # Network & Resilience
    reconnect_delay_seconds: float = 3.0
    ws_base_url: str | None = None
    open_timeout: float = 30.0
    close_timeout: float = 5.0
    ping_interval: float | None = None

    # Flat Audio Hardware Device Routing (for backwards compatibility)
    input_device: int | str | None = None
    output_device: int | str | None = None

    # Flat Sample Rates (for backwards compatibility)
    mic_capture_rate: int = 16000
    gemini_input_rate: int = 16000
    speaker_sample_rate: int = 24000

    # Flat VAD & Barge-in Interruption Thresholds (for backwards compatibility)
    interrupt_rms_threshold: int = 2500
    ai_interrupt_threshold: int = 2800
    noise_gate_threshold: int = 250
    hang_frames: int = 15
    vad_streak_trigger: int = 4
    pre_buffer_max_frames: int = 8

    # Flat Storage & Media
    media_directory: str = "audio_experiences"

    # Optional metadata or custom user parameters
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sync flat properties with audio config if explicitly provided
        if self.input_device is not None:
            self.audio.input_device = self.input_device
        else:
            self.input_device = self.audio.input_device

        if self.output_device is not None:
            self.audio.output_device = self.output_device
        else:
            self.output_device = self.audio.output_device

        if self.interrupt_rms_threshold != 2500:
            self.audio.interrupt_rms_threshold = self.interrupt_rms_threshold
        else:
            self.interrupt_rms_threshold = self.audio.interrupt_rms_threshold

        if self.noise_gate_threshold != 250:
            self.audio.noise_gate_threshold = self.noise_gate_threshold
        else:
            self.noise_gate_threshold = self.audio.noise_gate_threshold

        if self.media_directory != "audio_experiences":
            self.audio.media_directory = self.media_directory
        else:
            self.media_directory = self.audio.media_directory


__all__ = [
    "VoiceConfig",
    "AudioConfig",
    "RAGConfig",
    "WhatsAppConfig",
]
