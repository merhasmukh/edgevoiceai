"""
Qdrant Vector Database Store Adapter.
Supports local, on-premise, and cloud Qdrant instances.
"""

from __future__ import annotations

import os
from typing import Any
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.stores.base import BaseVectorStore

try:
    # pyright: ignore[reportMissingImports]
    from qdrant_client import AsyncQdrantClient, models as qmodels  # type: ignore[import-untyped, import-not-found]
    QDRANT_AVAILABLE = True
except ImportError:
    AsyncQdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]
    QDRANT_AVAILABLE = False


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector database store adapter.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        url: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.url = url or os.environ.get("QDRANT_URL", f"http://{host}:{port}")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"Qdrant ({self.collection_name})"

    def _get_client(self) -> Any:
        if not QDRANT_AVAILABLE or AsyncQdrantClient is None:
            return None
        if self._client is None:
            self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key, timeout=5)
        return self._client

    async def is_available(self) -> bool:
        """Checks if Qdrant server is reachable and collection exists."""
        client = self._get_client()
        if client is None:
            return False
        try:
            collections = await client.get_collections()
            return any(c.name == self.collection_name for c in collections.collections)
        except Exception:
            return False

    async def search(
        self,
        query_vector: list[float],
        limit: int = 3,
        score_threshold: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Executes vector similarity search on Qdrant collection."""
        client = self._get_client()
        if client is None:
            return []

        q_filter = None
        if filter_metadata and qmodels is not None:
            conditions: list[Any] = [
                qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            q_filter = qmodels.Filter(must=conditions)

        try:
            # Use query_points API
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=q_filter,
                with_payload=True,
            )

            chunks: list[DocumentChunk] = []
            for hit in results.points:
                payload = hit.payload or {}
                chunk = DocumentChunk(
                    id=str(hit.id),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score if hit.score is not None else 0.0),
                    metadata=dict(payload.get("metadata", {})),
                    audio_url=payload.get("audio_url"),
                    vector_store=self.name,
                )
                chunks.append(chunk)

            return chunks
        except Exception as exc:
            print(f"⚠️  [QdrantVectorStore] Search error on {self.collection_name}: {exc}")
            return []

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """Upserts document chunks and embeddings into Qdrant."""
        client = self._get_client()
        if client is None or qmodels is None:
            raise RuntimeError("Qdrant client not available for upsert.")

        points = [
            qmodels.PointStruct(
                id=c.id,
                vector=v,
                payload={
                    "title": c.title,
                    "text": c.text,
                    "metadata": c.metadata,
                    "audio_url": c.audio_url,
                },
            )
            for c, v in zip(chunks, vectors)
        ]

        await client.upsert(collection_name=self.collection_name, points=points)
