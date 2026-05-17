# rag-core/rag_core/ingestion/pipeline.py
import re
from pathlib import Path
from rag_core.types import Chunk, IngestionResult, DEFAULT_NAMESPACE
from rag_core.capabilities.embedder import Embedder
from rag_core.capabilities.indexer import Indexer
from rag_core.hashing import hash_file
from rag_core.ingestion.loaders import get_loader
from rag_core.ingestion.splitter import create_splitter


def extract_metadata(documents: list) -> dict[str, str]:
    """Heuristic metadata extraction from document content.
    Extracts title, date, and author without heavy NLP dependencies.
    """
    meta: dict[str, str] = {}
    if not documents:
        return meta

    full_text = " ".join(
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in documents
    )

    # Title: first Markdown heading or first non-empty line
    title_match = re.search(r"^#\s+(.+)$", full_text, re.MULTILINE)
    if title_match:
        meta["title"] = title_match.group(1).strip()

    # Date: common formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
    date_match = re.search(
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
        full_text,
    )
    if date_match:
        meta["date"] = date_match.group(1)

    # Author: "Author: Name", "By Name", "Written by Name"
    author_match = re.search(
        r"(?:Author|By|Written by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        full_text, re.IGNORECASE,
    )
    if author_match:
        meta["author"] = author_match.group(1).strip()

    return meta


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
    doc_meta = extract_metadata(documents)

    splitter = create_splitter()
    raw_chunks = splitter.split_documents(documents)

    typed_chunks = [
        Chunk(
            id=f"{source_id}:{i}",
            content=c.page_content,
            metadata={
                **(c.metadata if hasattr(c, "metadata") else {}),
                **doc_meta,
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
