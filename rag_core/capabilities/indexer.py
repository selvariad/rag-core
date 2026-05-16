# rag-core/rag_core/capabilities/indexer.py
from typing import Protocol, runtime_checkable
from rag_core.types import Chunk, DEFAULT_NAMESPACE

__all__ = ["Indexer", "ChromaIndexer"]


@runtime_checkable
class Indexer(Protocol):
    async def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def delete(self, source_id: str, namespace: str = DEFAULT_NAMESPACE) -> None: ...
    async def source_exists(self, source_id: str, namespace: str = DEFAULT_NAMESPACE) -> bool: ...
    async def list_sources(self, namespace: str = DEFAULT_NAMESPACE) -> list[dict]: ...


class ChromaIndexer:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "documents"):
        import chromadb
        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    async def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[{**c.metadata, "source_id": c.source_id, "namespace": c.namespace}
                       for c in chunks],
            embeddings=vectors,
        )

    async def delete(self, source_id: str, namespace: str = DEFAULT_NAMESPACE) -> None:
        results = self._collection.get(
            where={"$and": [{"source_id": source_id}, {"namespace": namespace}]},
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    async def source_exists(self, source_id: str, namespace: str = DEFAULT_NAMESPACE) -> bool:
        results = self._collection.get(
            where={"$and": [{"source_id": source_id}, {"namespace": namespace}]},
            limit=1,
        )
        return len(results["ids"]) > 0

    async def list_sources(self, namespace: str = DEFAULT_NAMESPACE) -> list[dict]:
        results = self._collection.get(
            where={"namespace": namespace},
            include=["metadatas"],
        )
        seen: dict[str, dict] = {}
        if results["ids"]:
            for i, sid in enumerate(results["ids"]):
                source_id = results["metadatas"][i].get("source_id", "") if results["metadatas"] else ""
                if source_id and source_id not in seen:
                    meta = results["metadatas"][i] or {}
                    filename = meta.get("filename", source_id)
                    file_type = meta.get("file_type", "")
                    chunk_count = sum(1 for j in range(len(results["ids"]))
                                     if results["metadatas"][j].get("source_id") == source_id)
                    seen[source_id] = {
                        "source_id": source_id,
                        "filename": filename,
                        "file_type": file_type,
                        "chunk_count": chunk_count,
                    }
        return list(seen.values())
