# rag-core/rag_core/ingestion/pipeline.py
from pathlib import Path
from rag_core.types import Chunk, IngestionResult, DEFAULT_NAMESPACE
from rag_core.capabilities.embedder import Embedder
from rag_core.capabilities.indexer import Indexer
from rag_core.hashing import hash_file
from rag_core.ingestion.loaders import get_loader
from rag_core.ingestion.splitter import create_splitter


async def ingest(
    source: Path,
    indexer: Indexer,
    embedder: Embedder,
    namespace: str = DEFAULT_NAMESPACE,
    force: bool = False,
) -> IngestionResult:
    content_hash = hash_file(source)
    source_id = f"{namespace}:{content_hash}"

    if await indexer.source_exists(source_id, namespace):
        if not force:
            return IngestionResult.duplicate(source_id)
        await indexer.delete(source_id, namespace)

    loader = get_loader(source.suffix)
    documents = loader.load(source)

    splitter = create_splitter()
    raw_chunks = splitter.split_documents(documents)

    typed_chunks = [
        Chunk(
            id=f"{source_id}:{i}",
            content=c.page_content,
            metadata={
                **(c.metadata if hasattr(c, "metadata") else {}),
                "filename": source.name,
                "file_type": source.suffix,
            },
            source_id=source_id,
            namespace=namespace,
        )
        for i, c in enumerate(raw_chunks)
    ]

    if not typed_chunks:
        return IngestionResult(source_id=source_id, chunk_count=0, status="success")

    vectors = await embedder.embed([c.content for c in typed_chunks])
    await indexer.index(typed_chunks, vectors)

    return IngestionResult(
        source_id=source_id,
        chunk_count=len(typed_chunks),
        status="success",
    )
