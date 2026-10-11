# The conservation in UserA doesn't get into the
# convversation made by user B

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from .models import Message


@dataclass(slots=True)
class Interaction:
    type: Literal["user", "assistant", "tool"]
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    name: str | None = None
    tool_call_id: str | None = None

    def to_message(self) -> Message:
        return Message(self.type, self.content, name=self.name, tool_call_id=self.tool_call_id)


class MemoryStore(Protocol):
    async def add(self, session_id: str, interaction: Interaction) -> None: ...

    async def add_turn(self, session_id: str, user: str, assistant: str) -> None: ...

    async def add_tool(self, session_id: str, name: str, content: str, call_id: str) -> None: ...

    async def history(self, session_id: str) -> list[Interaction]: ...

    async def messages(self, session_id: str) -> list[Message]: ...

    async def clear(self, session_id: str) -> None: ...


class ConversationMemory:
    def __init__(self, max_history_length: int = 50):
        if max_history_length < 1:
            raise ValueError("max_history_length must be positive")
        self.max_history_length = max_history_length
        self._sessions: defaultdict[str, list[Interaction]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add(self, session_id: str, interaction: Interaction) -> None:
        async with self._lock:
            history = self._sessions[session_id]
            history.append(interaction)
            if len(history) > self.max_history_length:
                del history[: len(history) - self.max_history_length]

    async def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        await self.add(session_id, Interaction("user", user))
        await self.add(session_id, Interaction("assistant", assistant))

    async def add_tool(self, session_id: str, name: str, content: str, call_id: str) -> None:
        await self.add(session_id, Interaction("tool", content, name=name, tool_call_id=call_id))

    async def history(self, session_id: str) -> list[Interaction]:
        async with self._lock:
            return list(self._sessions.get(session_id, []))

    async def messages(self, session_id: str) -> list[Message]:
        return [item.to_message() for item in await self.history(session_id)]

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def session_count(self) -> int:
        async with self._lock:
            return len(self._sessions)
