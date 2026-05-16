# rag-core/rag_core/capabilities/retriever.py
from typing import Protocol, runtime_checkable
from rag_core.types import RetrievalQuery, RetrievedChunk

__all__ = ["Retriever", "HybridRetriever"]


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Returns chunks sorted by descending score."""
        ...


class HybridRetriever:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._use_reranker = use_reranker
        self._reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(reranker_model)
            except ImportError:
                pass

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        where_clause = {}
        if query.filters:
            where_clause = self._build_where(query.filters)
        where_clause["namespace"] = query.namespace

        results = self._collection.query(
            query_texts=[query.text],
            n_results=query.top_k * 2,  # oversample for rerank
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] or {}
                source_id = meta.pop("source_id", "")
                ns = meta.pop("namespace", query.namespace)
                chunks.append(RetrievedChunk(
                    id=chunk_id,
                    content=results["documents"][0][i] or "",
                    metadata=meta,
                    source_id=source_id,
                    namespace=ns,
                    score=1.0 - results["distances"][0][i],
                    rank=0,
                ))

        if self._reranker and len(chunks) > query.top_k:
            pairs = [[query.text, c.content] for c in chunks]
            scores = self._reranker.predict(pairs)
            for c, s in zip(chunks, scores):
                c.score = float(s)
            chunks.sort(key=lambda c: c.score, reverse=True)

        chunks = chunks[:query.top_k]
        for i, c in enumerate(chunks):
            c.rank = i + 1
        return chunks

    def _build_where(self, f) -> dict:
        from rag_core.types import MetadataFilter
        if f.and_:
            return {"$and": [self._build_where(sub) for sub in f.and_]}
        if f.or_:
            return {"$or": [self._build_where(sub) for sub in f.or_]}
        op_map = {
            "eq": "$eq", "ne": "$ne", "in": "$in", "gt": "$gt",
            "gte": "$gte", "lt": "$lt", "lte": "$lte", "contains": "$contains",
        }
        return {f.field: {op_map[f.op]: f.value}}
