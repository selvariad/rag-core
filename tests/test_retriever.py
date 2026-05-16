# rag-core/tests/test_retriever.py
import pytest
from rag_core.capabilities.retriever import Retriever
from rag_core.types import RetrievalQuery, RetrievedChunk, Chunk, MetadataFilter


class FakeRetriever:
    def __init__(self):
        self._store: dict[str, list[tuple[Chunk, list[float]]]] = {}

    async def index_chunks(self, chunks: list[Chunk], vectors: list[list[float]]):
        for c, v in zip(chunks, vectors):
            self._store.setdefault(c.namespace, []).append((c, v))

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        entries = self._store.get(query.namespace, [])
        results = []
        for i, (chunk, _) in enumerate(entries):
            if query.filters:
                f = query.filters
                if f.field and chunk.metadata.get(f.field) != f.value:
                    continue
            results.append(RetrievedChunk(
                id=chunk.id, content=chunk.content, metadata=chunk.metadata,
                source_id=chunk.source_id, namespace=chunk.namespace,
                score=1.0 - i * 0.1, rank=i + 1,
            ))
        return results[:query.top_k]


def test_fake_retriever_satisfies_protocol():
    r = FakeRetriever()
    assert isinstance(r, Retriever)


@pytest.mark.asyncio
async def test_retrieve_empty():
    r = FakeRetriever()
    result = await r.retrieve(RetrievalQuery(text="test"))
    assert result == []


@pytest.mark.asyncio
async def test_retrieve_basic():
    r = FakeRetriever()
    chunks = [
        Chunk(id="a:0", content="PTO policy details", metadata={"dept": "HR"}, source_id="a"),
        Chunk(id="a:1", content="Benefits overview", metadata={"dept": "HR"}, source_id="a"),
    ]
    await r.index_chunks(chunks, [[0.1]*8, [0.2]*8])
    result = await r.retrieve(RetrievalQuery(text="PTO"))
    assert len(result) == 2
    assert result[0].rank == 1
    assert result[0].score > result[1].score
    assert "PTO" in result[0].content


@pytest.mark.asyncio
async def test_retrieve_with_namespace_isolation():
    r = FakeRetriever()
    chunks_a = [Chunk(id="a:0", content="doc A", metadata={}, source_id="a", namespace="ns_a")]
    chunks_b = [Chunk(id="b:0", content="doc B", metadata={}, source_id="b", namespace="ns_b")]
    await r.index_chunks(chunks_a, [[0.1]*8])
    await r.index_chunks(chunks_b, [[0.2]*8])

    result = await r.retrieve(RetrievalQuery(text="doc", namespace="ns_a"))
    assert len(result) == 1
    assert result[0].content == "doc A"


@pytest.mark.asyncio
async def test_retrieve_with_metadata_filter():
    r = FakeRetriever()
    chunks = [
        Chunk(id="a:0", content="HR policy", metadata={"dept": "HR"}, source_id="a"),
        Chunk(id="a:1", content="Eng doc", metadata={"dept": "Eng"}, source_id="a"),
    ]
    await r.index_chunks(chunks, [[0.1]*8, [0.2]*8])
    result = await r.retrieve(RetrievalQuery(
        text="policy",
        filters=MetadataFilter(field="dept", op="eq", value="HR"),
    ))
    assert len(result) == 1
    assert result[0].content == "HR policy"


@pytest.mark.asyncio
async def test_retrieve_respects_top_k():
    r = FakeRetriever()
    chunks = [Chunk(id=f"a:{i}", content=f"doc {i}", metadata={}, source_id="a") for i in range(10)]
    await r.index_chunks(chunks, [[0.1]*8]*10)
    result = await r.retrieve(RetrievalQuery(text="doc", top_k=3))
    assert len(result) == 3
