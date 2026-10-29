"""
In-Memory & Local File Vector Store.
Provides pure-Python vector search with cosine similarity and keyword fallback.
Zero external services or Docker containers required. Ideal for edge devices and offline testing.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.stores.base import BaseVectorStore


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two dense vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store with optional JSON persistence.
    """

    def __init__(self, name: str = "In-Memory Store") -> None:
        self._name = name
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    @property
    def name(self) -> str:
        return self._name

    async def is_available(self) -> bool:
        return True

    async def get_stats(self) -> dict[str, Any]:
        """Returns statistics of in-memory store."""
        return {
            "status": "online",
            "name": self.name,
            "points_count": len(self._chunks),
            "vectors_count": len(self._vectors),
        }

    async def delete_points(self, point_ids: list[str]) -> int:
        """Deletes points by ID."""
        id_set = set(point_ids)
        new_chunks = []
        new_vectors = []
        deleted = 0
        for c, v in zip(self._chunks, self._vectors):
            if c.id in id_set:
                deleted += 1
            else:
                new_chunks.append(c)
                new_vectors.append(v)
        self._chunks = new_chunks
        self._vectors = new_vectors
        return deleted

    async def delete_by_filter(self, filter_metadata: dict[str, Any]) -> bool:
        """Deletes chunks matching filter."""
        new_chunks = []
        new_vectors = []
        for c, v in zip(self._chunks, self._vectors):
            match = True
            for k, val in filter_metadata.items():
                if c.metadata.get(k) != val:
                    match = False
                    break
            if not match:
                new_chunks.append(c)
                new_vectors.append(v)
        self._chunks = new_chunks
        self._vectors = new_vectors
        return True

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """Stores chunks with their corresponding vectors."""
        for c, v in zip(chunks, vectors):
            c.vector_store = self.name
            self._chunks.append(c)
            self._vectors.append(v)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 3,
        score_threshold: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Performs cosine similarity search against stored vectors."""
        if not self._chunks:
            return []

        # If query vector is empty or all zeros, return empty (or let keyword search handle)
        if not query_vector or all(v == 0.0 for v in query_vector):
            return []

        results: list[tuple[float, DocumentChunk]] = []

        for chunk, vec in zip(self._chunks, self._vectors):
            # Check metadata filter
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = cosine_similarity(query_vector, vec)
            if score >= score_threshold:
                match_chunk = DocumentChunk(
                    id=chunk.id,
                    title=chunk.title,
                    text=chunk.text,
                    score=round(score, 4),
                    metadata=chunk.metadata,
                    audio_url=chunk.audio_url,
                    vector_store=self.name,
                )
                results.append((score, match_chunk))

        # Sort descending by score
        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:limit]]

    def search_keyword(
        self,
        query: str,
        limit: int = 3,
    ) -> list[DocumentChunk]:
        """
        Fast token-based lexical search fallback when embeddings are unavailable.
        """
        if not query or not self._chunks:
            return []

        words = [w.lower() for w in query.strip().split() if len(w) > 1]
        if not words:
            return []

        scored: list[tuple[int, DocumentChunk]] = []
        for chunk in self._chunks:
            haystack = f"{chunk.title} {chunk.text}".lower()
            match_count = sum(1 for w in words if w in haystack)
            if match_count > 0:
                scored.append((match_count, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            DocumentChunk(
                id=c.id,
                title=c.title,
                text=c.text,
                score=round(score / max(len(words), 1), 2),
                metadata=c.metadata,
                audio_url=c.audio_url,
                vector_store=f"{self.name} (Keyword)",
            )
            for score, c in scored[:limit]
        ]

    @classmethod
    def from_json_file(cls, filepath: str) -> InMemoryVectorStore:
        """
        Loads document chunks and vectors from a structured JSON file.
        Accepts list of dictionaries with {title, text, metadata, audio_url, vector?}.
        """
        store = cls(name=f"JSON Store ({os.path.basename(filepath)})")
        if not os.path.isfile(filepath):
            return store

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("records", [])
            for item in items:
                chunk = DocumentChunk(
                    title=item.get("title", ""),
                    text=item.get("text", "") or item.get("content", ""),
                    metadata=item.get("metadata", {}),
                    audio_url=item.get("audio_url"),
                    vector_store=store.name,
                )
                vec = item.get("vector") or item.get("embedding") or []
                store._chunks.append(chunk)
                store._vectors.append(vec if isinstance(vec, list) else [])
        except Exception as exc:
            print(f"⚠️  [InMemoryVectorStore] Error loading {filepath}: {exc}")

        return store
