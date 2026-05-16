# rag-core/rag_core/memory/conversation.py
from rag_core.types import Message


class ConversationStore:
    def __init__(self, max_turns: int = 20):
        self._store: dict[str, list[Message]] = {}
        self._max_turns = max_turns

    def add(self, conversation_id: str, message: Message) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(message)
        if len(self._store[conversation_id]) > self._max_turns:
            self._store[conversation_id] = self._store[conversation_id][-self._max_turns:]

    def get(self, conversation_id: str, n: int | None = None) -> list[Message]:
        msgs = self._store.get(conversation_id, [])
        if n is not None:
            msgs = msgs[-n:]
        return list(msgs)

    def delete(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
