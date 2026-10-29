"""
RAG Retriever for Voice AI Framework.
Orchestrates dense embedding generation, vector store lookup,
and automated fallback to local/in-memory knowledge when networks are offline.
"""

from __future__ import annotations

from typing import Any, Callable
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.embeddings.base import BaseEmbeddingModel
from edgevoiceai.rag.stores.base import BaseVectorStore
from edgevoiceai.rag.stores.memory import InMemoryVectorStore


class RAGRetriever:
    """
    Modular RAG retrieval engine.
    Connects any embedding model with primary and fallback vector stores.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        fallback_store: BaseVectorStore | None = None,
        default_limit: int = 3,
        default_score_threshold: float = 0.45,
        query_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.fallback_store = fallback_store
        self.default_limit = default_limit
        self.default_score_threshold = default_score_threshold
        self.query_normalizer = query_normalizer

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Retrieves relevant document chunks for a natural language voice query.
        Falls back to fallback_store if primary store is unreachable or yields no matches.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        if self.query_normalizer is not None:
            try:
                clean_query = self.query_normalizer(clean_query)
            except Exception as norm_err:
                print(f"⚠️  [RAGRetriever] Query normalizer error: {norm_err}")

        lim = limit if limit is not None else self.default_limit
        thresh = score_threshold if score_threshold is not None else self.default_score_threshold

        # 1. Generate query embedding vector
        query_vector: list[float] = []
        try:
            query_vector = await self.embedding_model.embed_query(clean_query)
        except Exception as emb_err:
            print(f"⚠️  [RAGRetriever] Embedding generation error: {emb_err}")

        # 2. Search primary vector store
        matches: list[DocumentChunk] = []
        if query_vector:
            try:
                matches = await self.vector_store.search(
                    query_vector=query_vector,
                    limit=lim,
                    score_threshold=thresh,
                    filter_metadata=filter_metadata,
                )
            except Exception as store_err:
                print(f"⚠️  [RAGRetriever] Primary store error ({self.vector_store.name}): {store_err}")

        # 3. Fallback to secondary store if primary returned no matches
        if not matches and self.fallback_store is not None:
            print(f"ℹ️  [RAGRetriever] Querying fallback store: {self.fallback_store.name}")
            try:
                if query_vector:
                    matches = await self.fallback_store.search(
                        query_vector=query_vector,
                        limit=lim,
                        score_threshold=thresh,
                        filter_metadata=filter_metadata,
                    )
                # If still no matches and fallback is InMemoryVectorStore, try keyword search
                if not matches and isinstance(self.fallback_store, InMemoryVectorStore):
                    matches = self.fallback_store.search_keyword(clean_query, limit=lim)
            except Exception as fb_err:
                print(f"⚠️  [RAGRetriever] Fallback store error ({self.fallback_store.name}): {fb_err}")

        return matches

    def build_context_text(
        self,
        matches: list[DocumentChunk],
        header: str | None = None,
        footer: str | None = None,
    ) -> str:
        """Formats retrieved chunks into a standardized context block for LLM prompts."""
        if not matches:
            return ""
        blocks = [f"- [{m.title}]: {m.text}" for m in matches if m.text]
        content = "\n\n".join(blocks)
        parts = []
        if header:
            parts.append(header)
        parts.append(content)
        if footer:
            parts.append(footer)
        return "\n\n".join(parts)

    async def __call__(
        self,
        query: str,
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Callable interface directly compatible with KnowledgeBaseTool's retriever_func.
        Returns list of dicts.
        """
        chunks = await self.retrieve(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )
        return [c.to_dict() for c in chunks]
