"""
Base interface for Vector Database Stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from edgevoiceai.rag.chunk import DocumentChunk


class BaseVectorStore(ABC):
    """
    Abstract Base Class for vector storage and semantic retrieval backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the vector database backend (e.g. In-Memory, Qdrant, Pinecone)."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int = 3,
        score_threshold: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Searches the vector store using dense query vectors.
        Returns matched DocumentChunks ranked by descending similarity score.
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """Stores or updates document chunks with their dense vector embeddings."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Returns True if the backend is reachable and ready to serve queries."""
        pass
