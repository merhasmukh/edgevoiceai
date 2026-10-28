"""
Embedding models for RAG pipeline.
"""

from edgevoiceai.rag.embeddings.base import BaseEmbeddingModel
from edgevoiceai.rag.embeddings.gemini import GeminiEmbeddings

__all__ = [
    "BaseEmbeddingModel",
    "GeminiEmbeddings",
]
