# rag-core/tests/test_chat_model.py
import pytest
from rag_core.capabilities.chat_model import ChatModel, LangChainChatModel
from rag_core.types import Message, ToolCall


class FakeChatModel:
    async def invoke(self, messages: list[Message], tools=None) -> Message:
        return Message(role="assistant", content=f"Echo: {messages[-1].content}")

    async def stream(self, messages: list[Message]):
        for char in messages[-1].content:
            yield char


def test_fake_chat_model_satisfies_protocol():
    m = FakeChatModel()
    assert isinstance(m, ChatModel)


@pytest.mark.asyncio
async def test_fake_chat_model_invoke():
    m = FakeChatModel()
    result = await m.invoke([Message(role="user", content="hello")])
    assert result.role == "assistant"
    assert "Echo" in result.content


@pytest.mark.asyncio
async def test_fake_chat_model_stream():
    m = FakeChatModel()
    chars = []
    async for c in m.stream([Message(role="user", content="hi")]):
        chars.append(c)
    assert "".join(chars) == "hi"


def test_to_langchain_all_roles(monkeypatch):
    monkeypatch.setattr("rag_core.capabilities.chat_model.LangChainChatModel._build_llm", lambda self: None)
    model = LangChainChatModel(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi", tool_calls=[
            ToolCall(id="c1", name="search", args={"q": "test"})
        ]),
        Message(role="tool", content="result", tool_call_id="c1"),
    ]
    lc = model._to_langchain(messages)
    assert len(lc) == 4
    assert lc[0].type == "system"
    assert lc[1].type == "human"
    assert lc[2].type == "ai"
    assert lc[2].tool_calls[0]["name"] == "search"
    assert lc[3].type == "tool"
    assert lc[3].tool_call_id == "c1"
