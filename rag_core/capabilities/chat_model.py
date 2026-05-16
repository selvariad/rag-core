# rag-core/rag_core/capabilities/chat_model.py
from typing import Protocol, runtime_checkable, AsyncIterator
from rag_core.types import Message, ToolDef, ToolCall

__all__ = ["ChatModel", "LangChainChatModel"]


@runtime_checkable
class ChatModel(Protocol):
    async def invoke(self, messages: list[Message], tools: list[ToolDef] | None = None) -> Message: ...
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]: ...


class LangChainChatModel:
    """Adapts LangChain BaseChatModel to our ChatModel protocol."""
    def __init__(self, provider: str, model: str, **kwargs):
        self._provider = provider
        self._model_name = model
        self._kwargs = kwargs
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def _build_llm(self):
        if self._provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=self._model_name, **self._kwargs)
        elif self._provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=self._model_name, **self._kwargs)
        elif self._provider == "deepseek":
            from langchain_openai import ChatOpenAI
            kwargs = {k: v for k, v in self._kwargs.items() if v is not None}
            kwargs.setdefault("base_url", "https://api.deepseek.com/v1")
            return ChatOpenAI(model=self._model_name, **kwargs)
        elif self._provider == "zhipu":
            from langchain_openai import ChatOpenAI
            kwargs = {k: v for k, v in self._kwargs.items() if v is not None}
            kwargs.setdefault("base_url", "https://open.bigmodel.cn/api/paas/v4")
            return ChatOpenAI(model=self._model_name, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    async def invoke(self, messages: list[Message], tools: list[ToolDef] | None = None) -> Message:
        lc_messages = self._to_langchain(messages)
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters
            }} for t in tools]
        result = await self._get_llm().ainvoke(lc_messages, **kwargs)
        tool_calls = None
        if hasattr(result, "tool_calls") and result.tool_calls:
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], args=tc["args"])
                for tc in result.tool_calls
            ]
        return Message(
            role="assistant",
            content=result.content if isinstance(result.content, str) else "",
            tool_calls=tool_calls,
        )

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        lc_messages = self._to_langchain(messages)
        async for chunk in self._get_llm().astream(lc_messages):
            if chunk.content and isinstance(chunk.content, str):
                yield chunk.content

    def _to_langchain(self, messages: list[Message]) -> list:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        lc = []
        for m in messages:
            if m.role == "system":
                lc.append(SystemMessage(content=m.content))
            elif m.role == "user":
                lc.append(HumanMessage(content=m.content, name=m.name or None))
            elif m.role == "assistant":
                ai_kwargs = {}
                if m.tool_calls:
                    ai_kwargs["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "args": tc.args}
                        for tc in m.tool_calls
                    ]
                lc.append(AIMessage(content=m.content, name=m.name or None, **ai_kwargs))
            elif m.role == "tool":
                lc.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id or ""))
        return lc
