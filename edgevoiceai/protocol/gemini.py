"""
Gemini Live Bidirectional WebSocket Protocol Client.
Constructs handshake payloads, encodes audio streaming frames,
deserializes model turns, and handles tool call responses.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_GEMINI_WS_BASE_URL = (
    "wss://generativelanguage.googleapis.com"
    "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


@dataclass(slots=True)
class ParsedServerMessage:
    """Represents normalized content extracted from a Gemini Live server message."""
    audio_pcm_chunks: list[bytes] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turn_complete: bool = False
    interrupted: bool = False
    error: dict[str, Any] | None = None
    raw_message: dict[str, Any] = field(default_factory=dict)


class GeminiLiveProtocol:
    """
    Encoder and parser for the Google Gemini Live BidiGenerateContent protocol.
    """

    @staticmethod
    def get_ws_url(api_key: str, base_url: str = DEFAULT_GEMINI_WS_BASE_URL) -> str:
        """Constructs the full WebSocket URL containing authentication query param."""
        clean_key = (api_key or "").strip()
        return f"{base_url}?key={clean_key}"

    @staticmethod
    def build_setup_message(
        model: str = "gemini-3.1-flash-live-preview",
        system_instruction: str = "",
        voice_name: str = "Puck",
        tools_declarations: list[dict[str, Any]] | None = None,
        is_reconnect: bool = False,
        reconnect_instruction: str = "",
    ) -> dict[str, Any]:
        """
        Builds the initial setup handshake payload sent immediately after
        WebSocket connection opens.
        """
        full_instruction = system_instruction
        if is_reconnect and reconnect_instruction:
            full_instruction = f"{full_instruction}\n\n{reconnect_instruction}"

        setup_content: dict[str, Any] = {
            "model": f"models/{model}" if not model.startswith("models/") else model,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice_name}
                    }
                },
            },
            "systemInstruction": {
                "parts": [{"text": full_instruction}]
            },
        }

        if tools_declarations:
            setup_content["tools"] = tools_declarations

        return {"setup": setup_content}

    @staticmethod
    def encode_audio_chunk(pcm_16k_bytes: bytes) -> str:
        """
        Encodes a 16kHz mono PCM chunk into a Gemini Live realtimeInput JSON message.
        """
        b64_audio = base64.b64encode(pcm_16k_bytes).decode("ascii")
        return json.dumps({
            "realtimeInput": {
                "audio": {
                    "mimeType": "audio/pcm;rate=16000",
                    "data": b64_audio,
                }
            }
        })

    @staticmethod
    def encode_tool_responses(function_responses: list[dict[str, Any]]) -> str:
        """
        Encodes tool execution responses into the expected toolResponse JSON payload.
        """
        return json.dumps({
            "toolResponse": {
                "functionResponses": function_responses
            }
        })

    @staticmethod
    def parse_message(raw_json_str: str) -> ParsedServerMessage:
        """
        Parses incoming server messages into structured components.
        """
        try:
            msg = json.loads(raw_json_str)
        except Exception:
            return ParsedServerMessage()

        parsed = ParsedServerMessage(raw_message=msg)

        # 1. Error check
        if "error" in msg:
            parsed.error = msg["error"]
            return parsed

        sc = msg.get("serverContent", {})

        # 2. Interruption detection
        if sc.get("interrupted"):
            parsed.interrupted = True

        # 3. Turn complete
        if sc.get("turnComplete"):
            parsed.turn_complete = True

        # 4. Tool calls (can be at top level or inside serverContent)
        tool_call = msg.get("toolCall") or sc.get("toolCall")
        if tool_call:
            parsed.tool_calls = tool_call.get("functionCalls", [])

        # 5. Output Transcription (live model speech transcription)
        out_transcription = sc.get("outputTranscription", {})
        if isinstance(out_transcription, dict):
            ot_text = out_transcription.get("text", "")
            if ot_text:
                parsed.text_parts.append(ot_text)

        # 6. Model turn parts (Audio + Text)
        model_turn = sc.get("modelTurn", {})
        for part in model_turn.get("parts", []):
            inline_data = part.get("inlineData", {})
            if inline_data.get("mimeType", "").startswith("audio/"):
                b64_data = inline_data.get("data", "")
                if b64_data:
                    try:
                        pcm_bytes = base64.b64decode(b64_data)
                        parsed.audio_pcm_chunks.append(pcm_bytes)
                    except Exception:
                        pass

            text = part.get("text", "")
            if text:
                parsed.text_parts.append(text)

        return parsed
