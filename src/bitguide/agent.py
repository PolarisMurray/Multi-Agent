from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .contracts import LLmClient, Retriever, Tool
from .memory import ConversationMemory, MemoryStore
from .models import AgentRequest, AgentResponse, Message
from .tools import ToolManager

class Agent:
    def __init__ (
        self,
        agent_id : str,
        SystemPrompt: str,
        llm : LLmClient,
        tools : Iterable[Tool] = (),
        retriever: Retriever | None = None,
        max_iterations: int = 8,
        memory: MemoryStore | None = None,
    ):



    def _tool_schemas(self) -> list[dict[str, Any]]:


    async def run(self, request: AgentRequest) -> AgentResponse:



    def tool_name(self) -> list[str]:
        return self._tools.names()

    async def clear_seesion(self, session_id: str) -> None:
        await self._memory.clear(session_id)