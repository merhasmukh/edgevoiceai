"""
Voice AI Framework.
A modular, full-duplex conversational voice framework for edge devices and servers.
Supports Raspberry Pi, Linux, macOS, and Windows on Python 3.10+.
"""

from edgevoiceai.agent import VoiceAgent
from edgevoiceai.audio.device import AudioDeviceManager, AudioDeviceInfo
from edgevoiceai.audio.media import MediaAudioPlayer
from edgevoiceai.audio.player import AudioPlayer
from edgevoiceai.audio.stream import MicrophoneStream
from edgevoiceai.audio.vad import VADConfig, VoiceActivityDetector, compute_rms
from edgevoiceai.config import VoiceConfig
from edgevoiceai.guardrails.farewell import FarewellDetector
from edgevoiceai.guardrails.topic_guard import TopicGuard
from edgevoiceai.memory.extractor import EntityExtractor
from edgevoiceai.memory.message import ChatMessage, ConversationSession
from edgevoiceai.memory.stores.base import BaseMemoryStore
from edgevoiceai.memory.stores.memory import InMemoryMemoryStore
from edgevoiceai.memory.stores.sqlite import SQLiteMemoryStore
from edgevoiceai.notifications.whatsapp import (
    WhatsAppClient,
    WhatsAppConfig,
    WhatsAppNotificationTool,
)
from edgevoiceai.protocol.gemini import GeminiLiveProtocol, ParsedServerMessage
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.embeddings.base import BaseEmbeddingModel
from edgevoiceai.rag.embeddings.gemini import GeminiEmbeddings
from edgevoiceai.rag.retriever import RAGRetriever
from edgevoiceai.rag.stores.base import BaseVectorStore
from edgevoiceai.rag.stores.memory import InMemoryVectorStore
from edgevoiceai.rag.stores.pinecone import PineconeVectorStore
from edgevoiceai.rag.stores.qdrant import QdrantVectorStore
from edgevoiceai.tools.audio_tool import AudioPlaybackTool
from edgevoiceai.tools.base import BaseTool, ToolExecutionContext, ToolResult
from edgevoiceai.tools.notification_tool import (
    TwoTurnConfirmationNotificationTool,
    default_normalize_phone,
)
from edgevoiceai.docs import get_site_directory, mount_docs, serve_docs
from edgevoiceai.tools.rag_tool import KnowledgeBaseTool
from edgevoiceai.tools.registry import ToolRegistry
from edgevoiceai.tools.sensor_tool import SensorTelemetryTool

__version__ = "0.1.0"


__all__ = [
    # Core Agent & Config
    "VoiceAgent",
    "VoiceConfig",
    # Documentation & Mounting
    "mount_docs",
    "serve_docs",
    "get_site_directory",
    # Tools System
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "ToolExecutionContext",
    "KnowledgeBaseTool",
    "AudioPlaybackTool",
    "SensorTelemetryTool",
    "TwoTurnConfirmationNotificationTool",
    "default_normalize_phone",
    # Notifications & WhatsApp
    "WhatsAppConfig",
    "WhatsAppClient",
    "WhatsAppNotificationTool",

    # Audio Subsystem
    "AudioDeviceManager",
    "AudioDeviceInfo",
    "AudioPlayer",
    "MediaAudioPlayer",
    "MicrophoneStream",
    "VoiceActivityDetector",
    "VADConfig",
    "compute_rms",
    # Guardrails
    "TopicGuard",
    "FarewellDetector",
    # Memory & Context
    "ChatMessage",
    "ConversationSession",
    "BaseMemoryStore",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "EntityExtractor",
    # Protocol
    "GeminiLiveProtocol",
    "ParsedServerMessage",
    # RAG & Vector Stores
    "DocumentChunk",
    "BaseEmbeddingModel",
    "GeminiEmbeddings",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    "RAGRetriever",
]


