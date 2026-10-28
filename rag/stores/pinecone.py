"""
Pinecone Serverless Cloud Vector Database Store Adapter.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.stores.base import BaseVectorStore

try:
    # pyright: ignore[reportMissingImports]
    from pinecone import Pinecone  # type: ignore[import-untyped, import-not-found]
    PINECONE_AVAILABLE = True
except ImportError:
    Pinecone = None  # type: ignore[assignment]
    PINECONE_AVAILABLE = False


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone serverless cloud vector database store adapter.
    """

    def __init__(
        self,
        index_name: str = "knowledge-base",
        api_key: str | None = None,
        namespace: str = "",
    ) -> None:
        self.index_name = index_name
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        self.namespace = namespace
        self._pc: Any = None
        self._index: Any = None

    @property
    def name(self) -> str:
        return f"Pinecone ({self.index_name})"

    def _get_index(self) -> Any:
        if not PINECONE_AVAILABLE or Pinecone is None or not self.api_key:
            return None
        if self._index is None:
            # pyright: ignore[reportCallIssue]
            self._pc = Pinecone(api_key=self.api_key)  # type: ignore[call-arg]
            self._index = self._pc.Index(self.index_name)
        return self._index

    async def is_available(self) -> bool:
        """Verifies connection to Pinecone cloud index."""
        if not PINECONE_AVAILABLE or Pinecone is None or not self.api_key:
            return False

        def _check() -> bool:
            try:
                idx = self._get_index()
                if idx is None:
                    return False
                idx.describe_index_stats()
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_check)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 3,
        score_threshold: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Queries Pinecone index asynchronously via thread offload."""
        if not self.api_key:
            return []

        def _do_query() -> list[DocumentChunk]:
            idx = self._get_index()
            if idx is None:
                return []

            kwargs: dict[str, Any] = {
                "vector": query_vector,
                "top_k": max(1, int(limit)),
                "include_metadata": True,
            }
            if self.namespace:
                kwargs["namespace"] = self.namespace
            if filter_metadata:
                kwargs["filter"] = filter_metadata

            results = idx.query(**kwargs)
            chunks: list[DocumentChunk] = []

            for match in getattr(results, "matches", []):
                score = float(match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0))
                if score < score_threshold:
                    continue

                meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {}
                chunk_id = str(match.get("id", "") if isinstance(match, dict) else getattr(match, "id", ""))

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        title=str(meta.get("title", "")),
                        text=str(meta.get("text", "")),
                        score=score,
                        metadata=meta,
                        audio_url=meta.get("audio_url"),
                        vector_store=self.name,
                    )
                )

            return chunks

        try:
            return await asyncio.to_thread(_do_query)
        except Exception as exc:
            print(f"⚠️  [PineconeVectorStore] Search error on {self.index_name}: {exc}")
            return []

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """Upserts vectors and metadata into Pinecone."""
        def _do_upsert() -> None:
            idx = self._get_index()
            if idx is None:
                raise RuntimeError("Pinecone index not initialized.")

            records = [
                {
                    "id": c.id,
                    "values": v,
                    "metadata": {
                        "title": c.title,
                        "text": c.text,
                        "audio_url": c.audio_url or "",
                        **c.metadata,
                    },
                }
                for c, v in zip(chunks, vectors)
            ]
            idx.upsert(vectors=records, namespace=self.namespace)

        await asyncio.to_thread(_do_upsert)
