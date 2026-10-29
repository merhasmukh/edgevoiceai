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
            self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key, timeout=5, check_compatibility=False)
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

    async def init_collection(self, vector_size: int = 768, distance: str = "Cosine") -> bool:
        """Ensures the collection exists with the required vector dimension and distance metric."""
        client = self._get_client()
        if client is None or qmodels is None:
            return False
        try:
            collections = await client.get_collections()
            existing = [c.name for c in collections.collections]
            if self.collection_name not in existing:
                dist_metric = qmodels.Distance.COSINE
                if distance.lower() == "euclid":
                    dist_metric = qmodels.Distance.EUCLID
                elif distance.lower() == "dot":
                    dist_metric = qmodels.Distance.DOT

                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=dist_metric,
                    ),
                )
            return True
        except Exception as exc:
            print(f"⚠️  [QdrantVectorStore] Collection init skipped/error: {exc}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Retrieves collection statistics (point counts, indexed vectors, dimension)."""
        client = self._get_client()
        if client is None or not await self.is_available():
            return {
                "status": "offline",
                "collection_name": self.collection_name,
                "points_count": 0,
                "vectors_count": 0,
            }
        try:
            info = await client.get_collection(collection_name=self.collection_name)
            points = getattr(info, "points_count", 0) or 0
            vectors = getattr(info, "indexed_vectors_count", points) or points
            vector_size = 768
            distance = "Cosine"
            try:
                if hasattr(info.config.params, "vectors"):
                    v_params = info.config.params.vectors
                    if hasattr(v_params, "size"):
                        vector_size = v_params.size
                    if hasattr(v_params, "distance"):
                        distance = str(v_params.distance.value if hasattr(v_params.distance, "value") else v_params.distance)
            except Exception:
                pass

            return {
                "status": "online",
                "collection_name": self.collection_name,
                "points_count": points,
                "vectors_count": vectors,
                "vector_size": vector_size,
                "distance": distance,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "points_count": 0}

    async def delete_points(self, point_ids: list[str]) -> int:
        """Deletes specific point IDs from Qdrant."""
        if not point_ids or qmodels is None:
            return 0
        client = self._get_client()
        if client is None:
            return 0
        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
            return len(point_ids)
        except Exception as exc:
            print(f"⚠️  [QdrantVectorStore] Delete points error: {exc}")
            return 0

    async def delete_by_filter(self, filter_metadata: dict[str, Any]) -> bool:
        """Deletes all points matching the given metadata filter."""
        if not filter_metadata or qmodels is None:
            return False
        client = self._get_client()
        if client is None:
            return False
        try:
            conditions: list[Any] = [
                qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            await client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(must=conditions)
                ),
            )
            return True
        except Exception as exc:
            print(f"⚠️  [QdrantVectorStore] Delete by filter error: {exc}")
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
            must_conditions: list[Any] = []
            should_conditions: list[Any] = []
            for k, v in filter_metadata.items():
                if isinstance(v, list):
                    for item in v:
                        should_conditions.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=item)))
                else:
                    must_conditions.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)))

            if must_conditions and should_conditions:
                q_filter = qmodels.Filter(must=must_conditions, should=should_conditions)
            elif should_conditions:
                q_filter = qmodels.Filter(should=should_conditions)
            elif must_conditions:
                q_filter = qmodels.Filter(must=must_conditions)

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
                meta = dict(payload.get("metadata", {}))
                for k, v in payload.items():
                    if k not in ("title", "text", "metadata", "audio_url"):
                        meta[k] = v
                chunk = DocumentChunk(
                    id=str(hit.id),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score if hit.score is not None else 0.0),
                    metadata=meta,
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
        """Upserts document chunks and embeddings into Qdrant in batches."""
        client = self._get_client()
        if client is None or qmodels is None:
            raise RuntimeError("Qdrant client not available for upsert.")

        await self.init_collection()

        points = [
            qmodels.PointStruct(
                id=c.id,
                vector=v,
                payload={
                    "title": c.title,
                    "text": c.text,
                    "audio_url": c.audio_url or "",
                    "metadata": c.metadata,
                    **c.metadata,
                },
            )
            for c, v in zip(chunks, vectors)
        ]

        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(collection_name=self.collection_name, points=batch)
