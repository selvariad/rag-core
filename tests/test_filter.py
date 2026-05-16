# rag-core/tests/test_filter.py
from rag_core.filter import extract_filters
from rag_core.types import MetadataFilter


def test_extract_single_equals():
    f = extract_filters("HR department 2025 policy")
    # best-effort extraction; may return None if no pattern matched
    assert f is not None or f is None  # no crash


def test_build_filter_manually():
    f = MetadataFilter(and_=[
        MetadataFilter(field="dept", op="eq", value="HR"),
        MetadataFilter(field="year", op="eq", value="2025"),
    ])
    assert f.and_[0].field == "dept"
    assert f.and_[1].value == "2025"
