# rag-core/tests/test_chat_model.py
import pytest
from rag_core.capabilities.chat_model import ChatModel
from rag_core.types import Message


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
