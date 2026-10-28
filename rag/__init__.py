"""
RAG (Retrieval-Augmented Generation) & Vector Store Subsystem for Voice AI.
Supports In-Memory, Qdrant, Pinecone, and Gemini Embeddings on Python 3.10+.
"""

from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.embeddings.base import BaseEmbeddingModel
from edgevoiceai.rag.embeddings.gemini import GeminiEmbeddings
from edgevoiceai.rag.retriever import RAGRetriever
from edgevoiceai.rag.stores.base import BaseVectorStore
from edgevoiceai.rag.stores.memory import InMemoryVectorStore
from edgevoiceai.rag.stores.pinecone import PineconeVectorStore
from edgevoiceai.rag.stores.qdrant import QdrantVectorStore

__all__ = [
    "DocumentChunk",
    "BaseEmbeddingModel",
    "GeminiEmbeddings",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    "RAGRetriever",
]
