from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from .contracts import Closeable, LLMClient, Retriever, Tool
from .memory import ConversationMemory, Interaction, MemoryStore
from .models import AgentRequest, AgentResponse, Message
from .observability import NullTracer, Tracer
from .skills import SkillManager
from .tools import ToolManager


class Agent:
    """ReAct loop: ask the model, run the tools it requests, repeat until it answers."""

    def __init__(
        self,
        agent_id: str,
        system_prompt: str,
        llm: LLMClient,
        tools: Iterable[Tool] = (),
        retriever: Retriever | None = None,
        max_iterations: int = 8,
        memory: MemoryStore | None = None,
        *,
        name: str | None = None,
        description: str = "",
        skills: SkillManager | None = None,
        tracer: Tracer | None = None,
        resources: Iterable[Closeable] = (),
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.agent_id = agent_id
        self.name = name or agent_id
        self.description = description
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._llm = llm
        self._retriever = retriever
        self._memory = memory if memory is not None else ConversationMemory()
        self._skills = skills
        self._tracer = tracer or NullTracer()
        # Agent 拥有这些资源（LLM 连接、MCP 子进程、向量库），close() 时统一释放
        self._resources = list(resources)
        self._closed = False
        self._tools = ToolManager()
        for tool in tools:
            self._tools.register(tool)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def _system_message(self, query: str) -> Message:
        prompt = self.system_prompt
        if self._skills is not None:
            prompt, matched = self._skills.enhance_prompt(prompt, query)
            if matched:
                self._tracer.emit("skills.matched", self.agent_id, skills=[s.name for s in matched])
        if self._retriever is None:
            return Message("system", prompt)
        documents = await self._retriever.retrieve(query)
        self._tracer.emit("rag.retrieved", self.agent_id, count=len(documents))
        if not documents:
            return Message("system", prompt)
        context = "\n".join(f"- {doc}" for doc in documents)
        return Message("system", f"{prompt}\n\n参考资料：\n{context}")

    async def run(self, request: AgentRequest) -> AgentResponse:
        session_id = request.session_id
        user = Interaction("user", request.message)
        self._tracer.emit("agent.start", self.agent_id, session=session_id, message=request.message)
        # 只把 user / assistant 历史发给模型；工具记录留在记忆里做审计
        history = [m for m in await self._memory.messages(session_id) if m.role != "tool"]
        messages = [
            await self._system_message(request.message),
            *history,
            Message("user", request.message),
        ]
        schemas = self._tool_schemas()
        tool_records: list[Interaction] = []

        for iteration in range(1, self.max_iterations + 1):
            turn = await self._llm.complete(messages, schemas)
            self._tracer.emit(
                "llm.turn",
                self.agent_id,
                iteration=iteration,
                content=turn.content,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
            )

            if turn.tool_calls:
                messages.append(Message("assistant", turn.content, tool_calls=turn.tool_calls))
                for call in turn.tool_calls:
                    result = await self._tools.invoke(call.name, call.arguments)
                    content = result.for_model()
                    self._tracer.emit(
                        "tool.result",
                        self.agent_id,
                        name=call.name,
                        success=result.success,
                        content=content,
                    )
                    messages.append(Message("tool", content, name=call.name, tool_call_id=call.id))
                    tool_records.append(
                        Interaction("tool", content, name=call.name, tool_call_id=call.id)
                    )
                continue

            answer = turn.content.strip()
            if not answer:
                raise RuntimeError("Model returned an empty answer")

            # 成功后才按时间顺序一次性写入记忆，失败的请求不留下半截对话
            await self._memory.add(session_id, user)
            for record in tool_records:
                await self._memory.add(session_id, record)
            await self._memory.add(session_id, Interaction("assistant", answer))
            self._tracer.emit("agent.answer", self.agent_id, iterations=iteration, content=answer)

            return AgentResponse(
                content=answer,
                agent_id=self.agent_id,
                session_id=session_id,
                iterations=iteration,
            )

        raise RuntimeError(f"Agent did not finish within {self.max_iterations} iterations")

    def tool_names(self) -> list[str]:
        return self._tools.names()

    async def clear_session(self, session_id: str) -> None:
        await self._memory.clear(session_id)

    async def close(self) -> None:
        """释放资源；重复调用是安全的。一个资源关闭失败不影响其他资源。"""
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(
            *(resource.close() for resource in self._resources), return_exceptions=True
        )
