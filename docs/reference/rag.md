# RAG & Embeddings Reference

This section documents the Retrieval-Augmented Generation (RAG) pipeline, text chunking strategies, vector store backends, and embedding generators.

---

## RAGRetriever

::: edgevoiceai.rag.retriever
    options:
      show_root_heading: true
      members:
        - RAGRetriever

---

## TextChunker & Chunk

::: edgevoiceai.rag.chunk
    options:
      show_root_heading: true
      members:
        - TextChunker
        - Chunk

---

## Vector Stores

### BaseVectorStore

::: edgevoiceai.rag.stores.base
    options:
      show_root_heading: true
      members:
        - BaseVectorStore

### InMemoryVectorStore

::: edgevoiceai.rag.stores.memory
    options:
      show_root_heading: true
      members:
        - InMemoryVectorStore

### QdrantVectorStore

::: edgevoiceai.rag.stores.qdrant
    options:
      show_root_heading: true
      members:
        - QdrantVectorStore

### PineconeVectorStore

::: edgevoiceai.rag.stores.pinecone
    options:
      show_root_heading: true
      members:
        - PineconeVectorStore

---

## Embedding Services

::: edgevoiceai.rag.embeddings.gemini
    options:
      show_root_heading: true
      members:
        - GeminiEmbeddingService

