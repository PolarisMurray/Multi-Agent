from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass(slots = True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass(slots = True)
class Message:
    role: Literal ["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory = list)

@dataclass (slots = True)
class ModelTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory = list)

@dataclass(slots = True)
class AgentRequest:
    message: str
    session_id: str
    request_id: str | None = None

@dataclass(slots = True)
class AgentResponse:
    content: str
    agent_id: str
    session_id: str
    iterations: int = 1