"""Agent 注册表：按 id 管理 Agent，并保证“热替换不打断在途请求”。

核心机制（引用计数 + 优雅退役）：
    async with registry.acquire("tech_helper") as agent:   # 引用 +1
        await agent.run(...)                               # 用的是借到的那个实例
                                                           # 离开 with：引用 -1

    replace() 只是把字典里的指针换成新实例，旧实例标记 retired；
    新请求拿到新实例，旧请求继续用旧实例；旧实例引用归零时才真正 close()。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .agent import Agent


@dataclass(slots=True, frozen=True)
class AgentInfo:
    id: str
    name: str
    description: str
    version: int


@dataclass(slots=True)
class _Entry:
    agent: Agent
    version: int
    references: int = 0
    retired: bool = False
    # idle 被 set 表示“当前没有请求在用”
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.idle.set()


class AgentRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()
        # 正在等待退役的后台任务 → 对应条目（停机超时时要强制关闭这些 Agent）
        self._retiring: dict[asyncio.Task[None], _Entry] = {}

    async def register(self, agent: Agent) -> None:
        async with self._lock:
            if agent.agent_id in self._entries:
                raise ValueError(f"Agent already registered: {agent.agent_id}")
            self._entries[agent.agent_id] = _Entry(agent, version=1)

    async def replace(self, agent: Agent) -> int:
        """注册或替换，返回新版本号。替换是在锁内一步完成的，不存在“空窗期”。"""
        async with self._lock:
            old = self._entries.get(agent.agent_id)
            version = old.version + 1 if old else 1
            self._entries[agent.agent_id] = _Entry(agent, version)
            if old is not None:
                self._retire(old)
            return version

    async def remove(self, agent_id: str) -> bool:
        async with self._lock:
            entry = self._entries.pop(agent_id, None)
            if entry is None:
                return False
            self._retire(entry)
            return True

    def _retire(self, entry: _Entry) -> None:
        entry.retired = True

        async def close_when_idle() -> None:
            await entry.idle.wait()  # 等最后一个在途请求归还
            await entry.agent.close()

        # 放到后台任务里等，replace() 本身立刻返回，不被慢请求拖住
        task = asyncio.create_task(close_when_idle())
        self._retiring[task] = entry
        task.add_done_callback(lambda done: self._retiring.pop(done, None))

    @asynccontextmanager
    async def acquire(self, agent_id: str) -> AsyncIterator[Agent]:
        async with self._lock:
            entry = self._entries.get(agent_id)
            if entry is None:
                raise KeyError(f"Agent not found: {agent_id}")
            entry.references += 1
            entry.idle.clear()
        try:
            yield entry.agent
        finally:
            # finally：请求成功、抛异常、被取消都会归还引用，避免泄漏导致旧实例永远关不掉
            async with self._lock:
                entry.references -= 1
                if entry.references == 0:
                    entry.idle.set()

    async def ids(self) -> list[str]:
        async with self._lock:
            return sorted(self._entries)

    async def infos(self) -> list[AgentInfo]:
        async with self._lock:
            return [
                AgentInfo(key, e.agent.name, e.agent.description, e.version)
                for key, e in sorted(self._entries.items())
            ]

    async def close(self, grace_seconds: float = 5) -> None:
        """停机：退役所有 Agent，最多等 grace_seconds 秒，超时就强制关闭。"""
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            for entry in entries:
                self._retire(entry)
            retiring = dict(self._retiring)
        if not retiring:
            return
        _, pending = await asyncio.wait(retiring, timeout=grace_seconds)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # 有请求一直没归还（比如代码里漏了 async with），停机时也必须把资源关掉
        await asyncio.gather(*(retiring[t].agent.close() for t in pending), return_exceptions=True)
