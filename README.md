# rag-core

Lightweight Python library for document intelligence. Exposes narrow capability protocols — no framework abstractions.

## Architecture

Four protocols, each with default implementations:

| Protocol | Role | Default |
|----------|------|---------|
| `Retriever` | Read — semantic search | `HybridRetriever` (ChromaDB + BM25 + optional BGE-Reranker) |
| `Indexer` | Write — document storage | `ChromaIndexer` (ChromaDB-backed) |
| `Embedder` | Text → vectors | `LocalEmbedder` (BGE-M3) or `OpenAIEmbedder` |
| `ChatModel` | LLM invocation | `LangChainChatModel` (50+ providers via LangChain) |

Read (`Retriever`) and write (`Indexer`) are separate protocols — swap both implementations to switch vector stores (e.g. ChromaDB → Qdrant) without touching application code.

## Key Design Decisions

- **No `RAGEngine`** — the caller composes capabilities directly
- **No `VectorStore` protocol** — internal detail of Retriever/Indexer impls
- **No `Pipeline` abstraction** — `ingest()` is just a function
- All types are plain dataclasses — no inheritance hierarchies

## Install

```bash
pip install -e .
```

## Usage

```python
from rag_core.capabilities.embedder import LocalEmbedder
from rag_core.capabilities.indexer import ChromaIndexer
from rag_core.capabilities.retriever import HybridRetriever
from rag_core.ingestion.pipeline import ingest
from rag_core.types import RetrievalQuery
from pathlib import Path

embedder = LocalEmbedder()
indexer = ChromaIndexer()
retriever = HybridRetriever(embedder=embedder)

# Ingest
result = await ingest(Path("doc.pdf"), indexer, embedder)

# Query
chunks = await retriever.retrieve(RetrievalQuery(text="PTO policy"))
```
