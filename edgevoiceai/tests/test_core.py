"""
Unit tests for the modular Voice AI Framework core modules.
Tests audio VAD, hardware device manager, topic guardrails, farewell detector,
protocol encoders/parsers, and agent configuration.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from edgevoiceai import (
    AudioDeviceManager,
    FarewellDetector,
    GeminiLiveProtocol,
    TopicGuard,
    VADConfig,
    VoiceActivityDetector,
    VoiceAgent,
    VoiceConfig,
    compute_rms,
)
from edgevoiceai.tools.base import BaseTool, ToolExecutionContext, ToolResult


class DummyEchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "Echoes back the input"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "OBJECT", "properties": {"msg": {"type": "STRING"}}}

    async def execute(self, args: dict, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(status="success", data={"echo": args.get("msg", "")})


def test_compute_rms():
    # Empty audio
    assert compute_rms(b"") == 0.0

    # Silence (all zeros)
    silence = b"\x00\x00" * 100
    assert compute_rms(silence) == 0.0

    # Sine-wave-like / active PCM
    active_pcm = b"\x00\x10\x00\x20\x00\x30" * 50
    rms = compute_rms(active_pcm)
    assert rms > 0.0


def test_voice_activity_detector_onset_retention():
    cfg = VADConfig(noise_gate_threshold=500, hang_frames=2, pre_buffer_max_frames=3)
    vad = VoiceActivityDetector(cfg)

    # 1. Feed quiet chunks (below threshold) -> gate closed
    quiet = b"\x00\x01" * 100
    streaming, frames = vad.process_chunk(quiet)
    assert not streaming
    assert len(frames) == 0
    assert len(vad.pre_buffer) == 1

    # 2. Feed active chunk (above threshold) -> gate opens, flushes pre-buffer + current chunk
    loud = b"\x00\x50" * 100  # High RMS
    streaming, frames = vad.process_chunk(loud)
    assert streaming
    assert len(frames) == 2  # 1 onset frame from pre_buffer + 1 current loud frame


def test_barge_in_streak_detection():
    cfg = VADConfig(interrupt_rms_threshold=500, streak_trigger=3)
    vad = VoiceActivityDetector(cfg)

    loud = b"\x00\x50" * 100

    # Streak 1
    assert not vad.check_barge_in(loud)
    # Streak 2
    assert not vad.check_barge_in(loud)
    # Streak 3 -> Triggers barge-in!
    assert vad.check_barge_in(loud)


def test_audio_device_manager_matching():
    # Integer index resolution
    assert AudioDeviceManager.find_device_index(2) == 2
    assert AudioDeviceManager.find_device_index("3") == 3
    assert AudioDeviceManager.find_device_index(None) is None
    assert AudioDeviceManager.find_device_index("") is None


def test_topic_guardrail():
    guard = TopicGuard(
        domain_name="Automotive",
        allowed_keywords=["car", "engine", "brake", "oil", "tire"],
        blocked_keywords=["crypto", "bitcoin", "politics"],
    )

    # Allowed
    is_ok, _ = guard.is_allowed("How do I change my engine oil?")
    assert is_ok

    # Blocked by explicitly blocked keyword
    is_ok, msg = guard.is_allowed("Buy some bitcoin")
    assert not is_ok
    assert "Specialized AI assistant for Automotive" in msg or "Automotive" in msg

    # Blocked by allowed keyword restriction
    is_ok, msg = guard.is_allowed("What is the capital of France?")
    assert not is_ok


def test_farewell_detector_distinguishes_gratitude():
    detector = FarewellDetector()

    # True farewells
    assert detector.is_farewell("Good bye, talk to you later")
    assert detector.is_farewell("bye")
    assert detector.is_farewell("ચલો આવજો")
    assert detector.is_farewell("[SESSION_CLOSED]")

    # Gratitude only (MUST NOT trigger farewell)
    assert not detector.is_farewell("Thank you so much!")
    assert not detector.is_farewell("Thanks for the help")
    assert not detector.is_farewell("ઓકે આપનો ખૂબ ખૂબ આભાર")
    assert not detector.is_farewell("ધન્યવાદ")


def test_gemini_live_protocol_setup_message():
    url = GeminiLiveProtocol.get_ws_url("test_api_key_123")
    assert "key=test_api_key_123" in url

    msg = GeminiLiveProtocol.build_setup_message(
        model="gemini-2.0-flash-exp",
        system_instruction="Be helpful",
        voice_name="Aoede",
        tools_declarations=[{"name": "test_tool"}],
    )

    setup = msg["setup"]
    assert setup["model"] == "models/gemini-2.0-flash-exp"
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Aoede"
    assert setup["systemInstruction"]["parts"][0]["text"] == "Be helpful"
    assert len(setup["tools"]) == 1


def test_gemini_live_protocol_parser():
    sample_json = """{
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "Hello user"},
                    {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AAAA"}}
                ]
            },
            "turnComplete": true
        }
    }"""

    parsed = GeminiLiveProtocol.parse_message(sample_json)
    assert parsed.turn_complete
    assert "Hello user" in parsed.text_parts
    assert len(parsed.audio_pcm_chunks) == 1
    assert not parsed.interrupted


def test_voice_agent_initialization():
    cfg = VoiceConfig(
        api_key="mock_key",
        model="gemini-2.0-flash-exp",
        voice="Puck",
        system_instruction="Custom agent",
    )

    agent = VoiceAgent(cfg)
    agent.register_tool(DummyEchoTool())

    assert agent.tools.get("echo_tool") is not None
    assert agent.config.voice == "Puck"


def test_edgevoiceai_mount_docs():
    """Verifies that documentation mounting utility integrates seamlessly with FastAPI."""
    from fastapi import FastAPI
    from edgevoiceai import mount_docs, get_site_directory

    test_app = FastAPI()
    success = mount_docs(test_app, path="/docs")
    assert success is True
    site_path = get_site_directory()
    assert (site_path / "index.html").is_file()


def test_configuration_dataclasses():
    """Verifies AudioConfig, RAGConfig, WhatsAppConfig, and modular VoiceConfig synchronization."""
    from edgevoiceai import AudioConfig, RAGConfig, WhatsAppConfig

    audio = AudioConfig(input_device=2, output_device=3, interrupt_rms_threshold=3000)
    assert audio.input_device == 2
    assert audio.output_device == 3
    assert audio.interrupt_rms_threshold == 3000

    rag = RAGConfig(backend="pinecone", collection_name="test_kb", dimension=768)
    assert rag.backend == "pinecone"
    assert rag.dimension == 768

    wa = WhatsAppConfig(provider="meta", phone_number_id="12345", token="token_xyz")
    assert wa.provider == "meta"

    # Initialize VoiceConfig with modular sub-configs
    v_cfg = VoiceConfig(api_key="mock", audio=audio, rag=rag, whatsapp=wa)
    assert v_cfg.audio.input_device == 2
    assert v_cfg.input_device == 2  # sync check
    assert v_cfg.rag is not None
    assert v_cfg.rag.backend == "pinecone"
    assert v_cfg.whatsapp is not None
    assert v_cfg.whatsapp.phone_number_id == "12345"
