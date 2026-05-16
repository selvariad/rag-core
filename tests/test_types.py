# rag-core/tests/test_types.py
import json
from dataclasses import asdict
from rag_core.types import (
    MetadataFilter, RetrievalQuery, Chunk, RetrievedChunk,
    IngestionResult, QueryResult, ToolDef, ToolCall, Message,
    DEFAULT_NAMESPACE,
)

def test_metadata_filter_simple():
    f = MetadataFilter(field="dept", op="eq", value="HR")
    d = asdict(f)
    assert d == {"field": "dept", "op": "eq", "value": "HR", "and_": None, "or_": None}

def test_metadata_filter_compound_and():
    f = MetadataFilter(and_=[
        MetadataFilter(field="dept", op="eq", value="HR"),
        MetadataFilter(field="year", op="eq", value="2025"),
    ])
    d = asdict(f)
    assert d["field"] is None
    assert len(d["and_"]) == 2

def test_retrieval_query_defaults():
    q = RetrievalQuery(text="PTO policy")
    assert q.text == "PTO policy"
    assert q.namespace == "default"
    assert q.filters is None
    assert q.top_k == 10

def test_retrieval_query_with_filters():
    q = RetrievalQuery(text="benefits", filters=MetadataFilter(field="dept", op="eq", value="HR"), top_k=5)
    assert q.filters.field == "dept"
    assert q.top_k == 5

def test_chunk_default_namespace():
    c = Chunk(id="a", content="text", metadata={}, source_id="src1")
    assert c.namespace == "default"

def test_retrieved_chunk_is_chunk():
    rc = RetrievedChunk(id="a", content="text", metadata={}, source_id="src1", score=0.95, rank=1)
    assert isinstance(rc, Chunk)
    assert rc.score == 0.95

def test_ingestion_result_success():
    r = IngestionResult(source_id="ns:abc123", chunk_count=12, status="success")
    assert r.status == "success"

def test_ingestion_result_duplicate_factory():
    r = IngestionResult.duplicate("ns:abc123")
    assert r.status == "duplicate"
    assert r.chunk_count == 0

def test_query_result_roundtrip():
    rc = RetrievedChunk(id="c1", content="hello", metadata={}, source_id="s1", score=0.9, rank=1)
    qr = QueryResult(answer="world", sources=[rc], trace_id="t1", tokens={"prompt": 100, "completion": 50})
    d = asdict(qr)
    assert d["answer"] == "world"
    assert len(d["sources"]) == 1
    assert d["tokens"]["prompt"] == 100

def test_message_tool_role_with_tool_call_id():
    m = Message(role="tool", content="42", tool_call_id="call_001")
    assert m.tool_call_id == "call_001"

def test_message_with_tool_calls():
    tc = ToolCall(id="c1", name="search", args={"q": "test"})
    m = Message(role="assistant", content="", tool_calls=[tc])
    assert m.tool_calls[0].name == "search"

def test_tool_def():
    td = ToolDef(name="search", description="Search documents", parameters={"type": "object", "properties": {}})
    d = asdict(td)
    assert d["name"] == "search"

def test_metadata_serialization_on_message():
    m = Message(role="user", content="hi", metadata={"trace_id": "abc"}, name="alice")
    d = asdict(m)
    assert d["metadata"]["trace_id"] == "abc"
    assert d["name"] == "alice"

def test_namespace_explicit_override():
    c = Chunk(id="a", content="x", metadata={}, source_id="s", namespace="tenant_a")
    assert c.namespace == "tenant_a"
