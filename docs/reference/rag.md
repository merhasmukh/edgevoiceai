# RAG & Embeddings Reference

This section documents the Retrieval-Augmented Generation (RAG) pipeline, document chunk representation, vector store backends, and embedding generators.

---

## DocumentChunk

::: edgevoiceai.rag.chunk
    options:
      show_root_heading: true
      members:
        - DocumentChunk

---

## RAGRetriever

::: edgevoiceai.rag.retriever
    options:
      show_root_heading: true
      members:
        - RAGRetriever

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

### BaseEmbeddingModel

::: edgevoiceai.rag.embeddings.base
    options:
      show_root_heading: true
      members:
        - BaseEmbeddingModel

### GeminiEmbeddings

::: edgevoiceai.rag.embeddings.gemini
    options:
      show_root_heading: true
      members:
        - GeminiEmbeddings
