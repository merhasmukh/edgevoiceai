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

    async def get_stats(self) -> dict[str, Any]:
        """Returns collection/index statistics (point counts, dimensions, status)."""
        available = await self.is_available()
        return {
            "status": "online" if available else "offline",
            "name": self.name,
            "points_count": 0,
            "vectors_count": 0,
        }

    async def delete_points(self, point_ids: list[str]) -> int:
        """Deletes specific point IDs from the vector store."""
        return 0

    async def delete_by_filter(self, filter_metadata: dict[str, Any]) -> bool:
        """Deletes documents matching the specified metadata filter."""
        return False

    async def init_collection(self) -> bool:
        """Ensures the collection or index exists, creating it if required."""
        return True
