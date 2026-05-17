# rag-core

Lightweight Python library for knowledge access. Exposes narrow capability protocols — no framework abstractions, no Engine class, no Pipeline.

## Architecture

**Five protocols**, each with default implementations:

| Protocol | Role | Default |
|----------|------|---------|
| `Retriever` | Semantic search (read) | `HybridRetriever` (ChromaDB vector + BM25 keyword + optional BGE-Reranker) |
| `Indexer` | Document storage (write) | `ChromaIndexer` (ChromaDB-backed) |
| `Embedder` | Text → vectors | `LocalEmbedder` (BGE-M3) or `OpenAIEmbedder` |
| `ChatModel` | LLM invocation | `LangChainChatModel` (50+ providers via LangChain) |
| `StructuredQueryEngine` | Read-only SQL execution | `SQLiteQueryEngine` (SELECT-only, table/column allowlists, default LIMIT) |

Read and write are separate protocols — swap implementations to switch vector stores (ChromaDB → Qdrant) without touching application code.

## Safety

- `SQLiteQueryEngine` enforces SELECT-only (no INSERT/DROP/PRAGMA)
- Table allowlist (`None`=all, `[]`=deny all, `["x"]`=whitelist)
- Column allowlist per table
- Default LIMIT applied automatically, `limit_applied` flag in result

## Design Decisions

- **No `RAGEngine`** — the caller composes capabilities directly
- **No `VectorStore` protocol** — internal detail of Retriever/Indexer impls
- **No `Pipeline` abstraction** — `ingest()` is just a function
- All types are plain dataclasses — no inheritance hierarchies (except `RetrievedChunk(Chunk)`)
- `KnowledgeRoute` type for multi-route dispatch (production_rag, structured_query, deep_research, etc.)

## Install

```bash
pip install -e .                    # core only
pip install -e ".[all]"             # everything
pip install -e ".[local-embedding,openai,ingestion]"  # pick what you need
```

Extras: `local-embedding` (BGE-M3), `openai` (OpenAI + LangChain OpenAI), `anthropic` (Claude), `ingestion` (PDF/Word/MD/HTML loaders).

## Usage

```python
from rag_core.ingestion.pipeline import ingest
from rag_core.capabilities.retriever import HybridRetriever
from rag_core.capabilities.structured_query import SQLiteQueryEngine
from rag_core.types import RetrievalQuery, StructuredQuery
from pathlib import Path

# Document Q&A
chunks = await retriever.retrieve(RetrievalQuery(text="PTO policy"))

# Structured data
engine = SQLiteQueryEngine(table_allowlist=["users"])
engine.setup_schema("CREATE TABLE users (id INTEGER, name TEXT)")
result = await engine.execute_readonly(StructuredQuery(sql="SELECT COUNT(*) FROM users"))
```
