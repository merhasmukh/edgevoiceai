"""
Real-time Voice AI Model Protocol Subsystem.
"""

from edgevoiceai.protocol.gemini import (
    DEFAULT_GEMINI_WS_BASE_URL,
    GeminiLiveProtocol,
    ParsedServerMessage,
)

__all__ = [
    "DEFAULT_GEMINI_WS_BASE_URL",
    "GeminiLiveProtocol",
    "ParsedServerMessage",
]
