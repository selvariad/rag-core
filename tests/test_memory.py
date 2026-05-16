# rag-core/tests/test_memory.py
from rag_core.memory.conversation import ConversationStore
from rag_core.types import Message


def test_add_and_get():
    store = ConversationStore(max_turns=10)
    store.add("conv1", Message(role="user", content="hello"))
    store.add("conv1", Message(role="assistant", content="hi there"))
    msgs = store.get("conv1", n=5)
    assert len(msgs) == 2


def test_get_respects_n():
    store = ConversationStore(max_turns=10)
    for i in range(5):
        store.add("c1", Message(role="user", content=str(i)))
    msgs = store.get("c1", n=2)
    assert len(msgs) == 2
    assert msgs[0].content == "3"
    assert msgs[1].content == "4"


def test_delete():
    store = ConversationStore(max_turns=10)
    store.add("c1", Message(role="user", content="hi"))
    store.delete("c1")
    assert store.get("c1") == []


def test_separate_conversations():
    store = ConversationStore(max_turns=10)
    store.add("a", Message(role="user", content="a"))
    store.add("b", Message(role="user", content="b"))
    assert len(store.get("a")) == 1
    assert store.get("a")[0].content == "a"
