# rag-core/rag_core/capabilities/embedder.py
from typing import Protocol, runtime_checkable
import asyncio

__all__ = ["Embedder", "LocalEmbedder", "OpenAIEmbedder"]


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LocalEmbedder. "
                "Install with: pip install sentence-transformers"
            )
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{model_name}': {exc}"
            ) from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True)
        )
        return [e.tolist() for e in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                text, normalize_embeddings=True, prompt_name="query"
            ),
        )
        return embedding.tolist()


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        resp = await client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]
