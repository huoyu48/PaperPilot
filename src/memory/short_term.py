"""Short-term memory: sliding window conversation buffer."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from src.utils.config import get_config
from src.utils.logging import logger


class ShortTermMemory:
    """Keeps the last K turns of conversation context."""

    def __init__(self, max_turns: int | None = None):
        cfg = get_config()
        self._max_turns = max_turns or cfg.max_memory_turns
        self._messages: list[BaseMessage] = []

    def add(self, messages: list[BaseMessage]) -> None:
        """Append messages and trim if exceeding max_turns."""
        self._messages.extend(messages)
        # Keep last max_turns * 2 messages (human + assistant pairs)
        max_msgs = self._max_turns * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]
        logger.debug(f"ShortTermMemory: {len(self._messages)} messages buffered")

    def get(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def to_context_string(self) -> str:
        """Format recent conversation as a string for LLM context injection."""
        parts: list[str] = []
        for msg in self._messages[-6:]:  # Last 3 exchanges
            role = "User" if msg.type == "human" else "Assistant"
            parts.append(f"{role}: {msg.content[:200]}")
        return "\n".join(parts)
