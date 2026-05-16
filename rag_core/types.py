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

    def __post_init__(self):
        is_simple = self.field is not None
        is_compound = self.and_ is not None or self.or_ is not None
        if is_simple and is_compound:
            raise ValueError(
                "Cannot set both simple (field/op/value) and compound (and_/or_) filter conditions"
            )
        if not is_simple and not is_compound:
            raise ValueError(
                "Filter must have either field/op/value or and_/or_"
            )


@dataclass
class RetrievalQuery:
    """Self-contained query value object. Not a raw string."""
    text: str
    namespace: str = DEFAULT_NAMESPACE
    filters: MetadataFilter | None = None
    top_k: int = 10

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


@dataclass(kw_only=True)
class Chunk:
    id: str
    content: str
    metadata: dict[str, Any]
    source_id: str
    namespace: str = DEFAULT_NAMESPACE


@dataclass(kw_only=True)
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
