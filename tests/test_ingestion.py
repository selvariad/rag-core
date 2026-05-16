# rag-core/tests/test_ingestion.py
import tempfile
import os
import pytest
from pathlib import Path

from rag_core.capabilities.embedder import Embedder
from rag_core.capabilities.indexer import Indexer
from rag_core.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_txt(content: str = "This is a test document about PTO policies.") -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def create_test_md(content: str = "# Test\n\nSome markdown content about benefits.") -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * 8


class FakeIndexer:
    def __init__(self):
        self._chunks: dict[str, list[Chunk]] = {}

    async def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for c in chunks:
            self._chunks.setdefault(c.source_id, []).append(c)

    async def delete(self, source_id: str, namespace: str = "default") -> None:
        self._chunks.pop(source_id, None)

    async def source_exists(self, source_id: str, namespace: str = "default") -> bool:
        return source_id in self._chunks

    async def list_sources(self, namespace: str = "default") -> list[dict]:
        return [{"source_id": sid, "filename": sid, "file_type": "", "chunk_count": len(chunks)}
                for sid, chunks in self._chunks.items()]


# ---------------------------------------------------------------------------
# get_loader tests
# ---------------------------------------------------------------------------

def test_get_loader_txt():
    from rag_core.ingestion.loaders import get_loader
    loader = get_loader(".txt")
    assert loader is not None
    assert hasattr(loader, "load")


def test_get_loader_md():
    from rag_core.ingestion.loaders import get_loader
    loader = get_loader(".md")
    assert loader is not None
    assert hasattr(loader, "load")


def test_get_loader_pdf():
    from rag_core.ingestion.loaders import get_loader
    loader = get_loader(".pdf")
    assert loader is not None
    assert hasattr(loader, "load")


def test_get_loader_unknown():
    from rag_core.ingestion.loaders import get_loader
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_loader(".xyz")


def test_get_loader_case_insensitive():
    from rag_core.ingestion.loaders import get_loader
    loader = get_loader(".TXT")
    assert loader is not None
    assert hasattr(loader, "load")


# ---------------------------------------------------------------------------
# ingest tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_txt_creates_chunks():
    from rag_core.ingestion.pipeline import ingest

    path = create_test_txt("This is a test document about PTO policies. " * 20)
    try:
        indexer = FakeIndexer()
        embedder = FakeEmbedder()
        result = await ingest(Path(path), indexer, embedder)

        assert result.status == "success"
        assert result.chunk_count > 0
        assert indexer._chunks
        # Verify chunks have expected metadata
        for source_id, chunks in indexer._chunks.items():
            for c in chunks:
                assert c.metadata["filename"] == Path(path).name
                assert c.metadata["file_type"] == ".txt"
                assert c.namespace == "default"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_ingest_dedup_skips_existing():
    from rag_core.ingestion.pipeline import ingest

    path = create_test_txt("Duplicate test content. " * 10)
    try:
        indexer = FakeIndexer()
        embedder = FakeEmbedder()

        # First ingestion
        result1 = await ingest(Path(path), indexer, embedder)
        assert result1.status == "success"
        assert result1.chunk_count > 0

        chunk_count_after_first = sum(len(v) for v in indexer._chunks.values())

        # Second ingestion — should be deduplicated
        result2 = await ingest(Path(path), indexer, embedder)
        assert result2.status == "duplicate"
        assert result2.chunk_count == 0

        # Chunk count should be unchanged
        chunk_count_after_second = sum(len(v) for v in indexer._chunks.values())
        assert chunk_count_after_second == chunk_count_after_first
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_ingest_md():
    from rag_core.ingestion.pipeline import ingest

    path = create_test_md("# Test\n\nSome markdown content about benefits. " * 20)
    try:
        indexer = FakeIndexer()
        embedder = FakeEmbedder()
        result = await ingest(Path(path), indexer, embedder)

        assert result.status == "success"
        assert result.chunk_count > 0
        # Verify chunks have expected metadata
        for source_id, chunks in indexer._chunks.items():
            for c in chunks:
                assert c.metadata["filename"] == Path(path).name
                assert c.metadata["file_type"] == ".md"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_ingest_force_reingest():
    from rag_core.ingestion.pipeline import ingest

    path = create_test_txt("Force re-ingest test. " * 15)
    try:
        indexer = FakeIndexer()
        embedder = FakeEmbedder()

        # First ingestion
        result1 = await ingest(Path(path), indexer, embedder)
        assert result1.status == "success"

        # Force re-ingestion
        result2 = await ingest(Path(path), indexer, embedder, force=True)
        assert result2.status == "success"
        assert result2.chunk_count > 0
    finally:
        os.unlink(path)
