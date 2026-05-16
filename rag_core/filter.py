# rag-core/rag_core/filter.py
from rag_core.types import MetadataFilter


def extract_filters(query_text: str) -> MetadataFilter | None:
    """Best-effort metadata filter extraction from natural language query.
    Returns None if no structured filters could be extracted.
    """
    # Heuristic extraction — MVP returns None
    return None


def merge_filters(a: MetadataFilter | None, b: MetadataFilter | None) -> MetadataFilter | None:
    if a is None:
        return b
    if b is None:
        return a
    return MetadataFilter(and_=[a, b])
