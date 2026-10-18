from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .models import Message, ModelTurn
from .tools import ToolResult


class LLMClient(Protocol):
    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn: ...


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult: ...


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[str]: ...


@runtime_checkable
class Closeable(Protocol):
    async def close(self) -> None: ...
