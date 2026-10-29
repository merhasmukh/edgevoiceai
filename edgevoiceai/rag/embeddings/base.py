"""
Base interface for RAG embedding models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """
    Abstract Base Class for text embedding generators.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generates a dense vector embedding for a single text document."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generates a dense vector embedding for a search query."""
        pass

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch generates embeddings for multiple texts."""
        embeddings: list[list[float]] = []
        for t in texts:
            emb = await self.embed_text(t)
            embeddings.append(emb)
        return embeddings

    async def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Alias for batched text embeddings generation."""
        return await self.embed_texts(texts)
