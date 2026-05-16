# rag-core/tests/test_embedder.py
import pytest
from rag_core.capabilities.embedder import Embedder, LocalEmbedder


class FakeEmbedder:
    """Embedder that returns fixed-dimension vectors for testing."""
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * 8


def test_fake_embedder_satisfies_protocol():
    e = FakeEmbedder()
    assert isinstance(e, Embedder)


@pytest.mark.asyncio
async def test_fake_embedder_returns_correct_shape():
    e = FakeEmbedder()
    result = await e.embed(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 8


@pytest.mark.asyncio
async def test_fake_embedder_query_returns_correct_shape():
    e = FakeEmbedder()
    result = await e.embed_query("hello")
    assert len(result) == 8


@pytest.mark.asyncio
async def test_local_embedder_satisfies_protocol():
    e = LocalEmbedder(model_name="BAAI/bge-small-en")
    assert isinstance(e, Embedder)


@pytest.mark.asyncio
async def test_local_embedder_embed():
    e = LocalEmbedder(model_name="BAAI/bge-small-en")
    result = await e.embed(["test document"])
    assert len(result) == 1
    assert len(result[0]) == 384  # bge-small-en dim


@pytest.mark.asyncio
async def test_local_embedder_embed_query():
    e = LocalEmbedder(model_name="BAAI/bge-small-en")
    result = await e.embed_query("test query")
    assert len(result) == 384
