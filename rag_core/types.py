# rag-core/rag_core/types.py
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_NAMESPACE = "default"


@dataclass
class MetadataFilter:
    """{field: "department", op: "eq", value: "HR"} or {and_: [...], or_: [...]}"""
    field: str | None = None
    op: Literal["eq", "ne", "in", "gt", "gte", "lt", "lte", "contains"] | None = None
    value: Any = None
    and_: list["MetadataFilter"] | None = None
    or_: list["MetadataFilter"] | None = None


@dataclass
class RetrievalQuery:
    """Self-contained query value object. Not a raw string."""
    text: str
    namespace: str = DEFAULT_NAMESPACE
    filters: MetadataFilter | None = None
    top_k: int = 10


@dataclass(kw_only=True)
class Chunk:
    id: str
    content: str
    metadata: dict[str, Any]
    source_id: str
    namespace: str = DEFAULT_NAMESPACE


@dataclass
class RetrievedChunk(Chunk):
    score: float
    rank: int


@dataclass
class IngestionResult:
    source_id: str
    chunk_count: int
    status: Literal["success", "duplicate", "updated"]

    @classmethod
    def duplicate(cls, source_id: str) -> "IngestionResult":
        return cls(source_id=source_id, chunk_count=0, status="duplicate")


@dataclass
class QueryResult:
    answer: str
    sources: list[RetrievedChunk]
    trace_id: str
    tokens: dict[str, int]


@dataclass
class ToolDef:
    """JSON Schema for a tool the model can call."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
