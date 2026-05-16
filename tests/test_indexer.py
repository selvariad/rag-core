# rag-core/tests/test_indexer.py
import pytest
from rag_core.capabilities.indexer import Indexer
from rag_core.types import Chunk


class FakeIndexer:
    def __init__(self):
        self._chunks: dict[str, list[Chunk]] = {}
        self._vectors: dict[str, list[list[float]]] = {}

    async def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for c in chunks:
            self._chunks.setdefault(c.source_id, []).append(c)
        self._vectors.setdefault(chunks[0].source_id, []).extend(vectors)

    async def delete(self, source_id: str, namespace: str = "default") -> None:
        self._chunks.pop(source_id, None)
        self._vectors.pop(source_id, None)

    async def source_exists(self, source_id: str, namespace: str = "default") -> bool:
        return source_id in self._chunks

    async def list_sources(self, namespace: str = "default") -> list[dict]:
        return [{"source_id": sid, "filename": sid, "file_type": "", "chunk_count": len(chunks)}
                for sid, chunks in self._chunks.items()]


def test_fake_indexer_satisfies_protocol():
    idx = FakeIndexer()
    assert isinstance(idx, Indexer)


@pytest.mark.asyncio
async def test_index_and_exists():
    idx = FakeIndexer()
    chunks = [Chunk(id="a:0", content="hello", metadata={}, source_id="a")]
    await idx.index(chunks, [[0.1, 0.2]])
    assert await idx.source_exists("a") is True
    assert await idx.source_exists("b") is False


@pytest.mark.asyncio
async def test_delete():
    idx = FakeIndexer()
    chunks = [Chunk(id="a:0", content="hello", metadata={}, source_id="a")]
    await idx.index(chunks, [[0.1, 0.2]])
    await idx.delete("a")
    assert await idx.source_exists("a") is False
