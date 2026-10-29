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
        self._avail_cache_value: bool | None = None
        self._avail_cache_ts: float = 0.0
        self._cache_ttl: float = 300.0

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

    def invalidate_cache(self) -> None:
        """Clears availability cache."""
        self._avail_cache_value = None
        self._avail_cache_ts = 0.0

    async def is_available(self) -> bool:
        """Verifies connection to Pinecone cloud index with cached result."""
        if not PINECONE_AVAILABLE or Pinecone is None or not self.api_key:
            return False

        import time
        now = time.monotonic()
        if self._avail_cache_value is not None and (now - self._avail_cache_ts) < self._cache_ttl:
            return self._avail_cache_value

        def _check() -> bool:
            try:
                idx = self._get_index()
                if idx is None:
                    return False
                idx.describe_index_stats()
                return True
            except Exception:
                return False

        res = await asyncio.to_thread(_check)
        self._avail_cache_value = res
        self._avail_cache_ts = now
        return res

    async def get_stats(self) -> dict[str, Any]:
        """Retrieves Pinecone index statistics."""
        if not self.api_key:
            return {"status": "offline", "index_name": self.index_name, "points_count": 0}

        def _do_stats() -> dict[str, Any]:
            idx = self._get_index()
            if idx is None:
                return {"status": "offline", "index_name": self.index_name, "points_count": 0}
            try:
                stats = idx.describe_index_stats()
                ns = stats.get("namespaces", {}).get(self.namespace, {})
                return {
                    "status": "online",
                    "index_name": self.index_name,
                    "namespace": self.namespace,
                    "vector_size": getattr(stats, "dimension", 768),
                    "points_count": ns.get("vector_count", 0),
                    "vectors_count": ns.get("vector_count", 0),
                    "total_vectors": stats.get("total_vector_count", 0),
                }
            except Exception as exc:
                return {"status": "error", "error": str(exc), "points_count": 0}

        return await asyncio.to_thread(_do_stats)

    async def delete_points(self, point_ids: list[str]) -> int:
        """Deletes specific vector IDs from Pinecone in batches."""
        if not point_ids or not self.api_key:
            return 0

        def _do_delete() -> int:
            idx = self._get_index()
            if idx is None:
                return 0
            batch_size = 100
            for i in range(0, len(point_ids), batch_size):
                batch = point_ids[i : i + batch_size]
                idx.delete(ids=batch, namespace=self.namespace)
            return len(point_ids)

        try:
            return await asyncio.to_thread(_do_delete)
        except Exception as exc:
            print(f"⚠️  [PineconeVectorStore] Delete points error: {exc}")
            return 0

    async def delete_by_filter(self, filter_metadata: dict[str, Any]) -> bool:
        """Deletes all vectors in Pinecone matching metadata filter."""
        if not filter_metadata or not self.api_key:
            return False

        def _do_delete_filter() -> bool:
            idx = self._get_index()
            if idx is None:
                return False
            idx.delete(filter=filter_metadata, namespace=self.namespace)
            return True

        try:
            return await asyncio.to_thread(_do_delete_filter)
        except Exception as exc:
            print(f"⚠️  [PineconeVectorStore] Delete by filter error: {exc}")
            return False

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
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                idx.upsert(vectors=batch, namespace=self.namespace)

        await asyncio.to_thread(_do_upsert)
