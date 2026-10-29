"""
Unit tests for the modular Voice AI Framework RAG & Vector Store subsystem.
Tests DocumentChunk, cosine similarity, InMemoryVectorStore, RAGRetriever,
fallback logic, and direct KnowledgeBaseTool integration.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.rag.embeddings.base import BaseEmbeddingModel
from edgevoiceai.rag.stores.memory import InMemoryVectorStore, cosine_similarity
from edgevoiceai.rag.retriever import RAGRetriever
from edgevoiceai.tools.rag_tool import KnowledgeBaseTool
from edgevoiceai.tools.base import ToolExecutionContext


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic mock embedding model for unit testing."""
    async def embed_text(self, text: str) -> list[float]:
        # Return simple deterministic vector based on character lengths
        val = float(len(text) % 10) / 10.0
        return [val, 1.0 - val, 0.5, 0.2]

    async def embed_query(self, query: str) -> list[float]:
        return await self.embed_text(query)


def test_document_chunk_serialization():
    chunk = DocumentChunk(
        title="Test Doc",
        text="Sample text content",
        score=0.89,
        metadata={"category": "test"},
        audio_url="/audio/sample.mp3",
    )
    d = chunk.to_dict()
    assert d["title"] == "Test Doc"
    assert d["score"] == 0.89
    assert d["audio_url"] == "/audio/sample.mp3"

    restored = DocumentChunk.from_dict(d)
    assert restored.title == chunk.title
    assert restored.score == chunk.score
    assert restored.metadata == chunk.metadata


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    v3 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)

    # Empty
    assert cosine_similarity([], []) == 0.0


@pytest.mark.asyncio
async def test_in_memory_vector_store():
    store = InMemoryVectorStore(name="Test Memory Store")
    chunks = [
        DocumentChunk(title="Python Guide", text="Learn Python programming"),
        DocumentChunk(title="Cooking Guide", text="How to cook pasta"),
    ]
    vectors = [
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.5],
    ]
    await store.upsert(chunks, vectors)

    # Search with vector close to Python Guide
    matches = await store.search([0.9, 0.1, 0.4, 0.0], limit=1, score_threshold=0.5)
    assert len(matches) == 1
    assert matches[0].title == "Python Guide"
    assert matches[0].score > 0.8


@pytest.mark.asyncio
async def test_in_memory_vector_store_keyword_fallback():
    store = InMemoryVectorStore()
    chunks = [
        DocumentChunk(title="Database Setup", text="Configure PostgreSQL and SQLite connection pool"),
        DocumentChunk(title="Audio Config", text="Microphone gain and noise gate levels"),
    ]
    await store.upsert(chunks, [[0.1, 0.2], [0.3, 0.4]])

    keyword_matches = store.search_keyword("PostgreSQL connection", limit=1)
    assert len(keyword_matches) == 1
    assert keyword_matches[0].title == "Database Setup"


@pytest.mark.asyncio
async def test_rag_retriever_orchestration_and_fallback():
    embedding = MockEmbeddingModel()
    primary_store = InMemoryVectorStore(name="Primary Empty Store")
    fallback_store = InMemoryVectorStore(name="Fallback Knowledge Store")

    fallback_chunks = [
        DocumentChunk(title="Reset Password", text="To reset your password, click forgot password link."),
    ]
    fallback_vectors = [
        await embedding.embed_text(fallback_chunks[0].text)
    ]
    await fallback_store.upsert(fallback_chunks, fallback_vectors)

    retriever = RAGRetriever(
        embedding_model=embedding,
        vector_store=primary_store,
        fallback_store=fallback_store,
        default_score_threshold=0.3,
    )

    # Primary has no records; retriever must automatically fallback and find Reset Password
    results = await retriever.retrieve("How do I reset my account password?", limit=1)
    assert len(results) == 1
    assert results[0].title == "Reset Password"

    ctx_text = retriever.build_context_text(results)
    assert "Reset Password" in ctx_text


@pytest.mark.asyncio
async def test_rag_retriever_callable_in_knowledge_base_tool():
    embedding = MockEmbeddingModel()
    store = InMemoryVectorStore()
    chunks = [
        DocumentChunk(title="Office Hours", text="Office is open Monday to Friday 9 AM to 5 PM."),
    ]
    vectors = [await embedding.embed_text(chunks[0].text)]
    await store.upsert(chunks, vectors)

    retriever = RAGRetriever(
        embedding_model=embedding,
        vector_store=store,
        default_score_threshold=0.3,
    )

    # Plug directly into KnowledgeBaseTool as retriever_func
    tool = KnowledgeBaseTool(
        name="search_office_info",
        description="Search office working hours and guidelines",
        parameters_schema={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}},
        retriever_func=retriever,
    )

    ctx = ToolExecutionContext(session_id="test_session")
    res = await tool.execute({"query": "When does the office open?"}, ctx)

    assert res.status == "success"
    assert "Office Hours" in res.data["results"]
    assert res.jit_instruction is not None


@pytest.mark.asyncio
async def test_vector_store_stats_and_delete_operations():
    store = InMemoryVectorStore(name="Test Operations Store")
    chunks = [
        DocumentChunk(id="c1", title="Doc 1", text="First doc", metadata={"author": "alice"}),
        DocumentChunk(id="c2", title="Doc 2", text="Second doc", metadata={"author": "bob"}),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    await store.upsert(chunks, vectors)

    stats = await store.get_stats()
    assert stats["status"] == "online"
    assert stats["points_count"] == 2

    # Delete point by ID
    deleted = await store.delete_points(["c1"])
    assert deleted == 1
    stats_after = await store.get_stats()
    assert stats_after["points_count"] == 1

    # Delete by filter
    await store.delete_by_filter({"author": "bob"})
    stats_final = await store.get_stats()
    assert stats_final["points_count"] == 0


@pytest.mark.asyncio
async def test_rag_retriever_query_normalizer_hook():
    embedding = MockEmbeddingModel()
    store = InMemoryVectorStore()
    chunks = [
        DocumentChunk(title="Billing FAQ", text="Invoices are issued on the 1st of every month."),
    ]
    vectors = [await embedding.embed_text(chunks[0].text)]
    await store.upsert(chunks, vectors)

    def mock_normalizer(q: str) -> str:
        return f"billing normalized {q}"

    retriever = RAGRetriever(
        embedding_model=embedding,
        vector_store=store,
        query_normalizer=mock_normalizer,
        default_score_threshold=0.1,
    )

    results = await retriever.retrieve("invoice query")
    assert len(results) == 1
    assert results[0].title == "Billing FAQ"
