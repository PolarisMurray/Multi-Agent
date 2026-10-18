"""工作流执行器：BSP（Bulk Synchronous Parallel，超步）模型。

每一“步”（superstep）分三段：
    1. 执行：frontier 里的节点并行执行（Semaphore 限制并发数），都读同一份 State 快照。
    2. 屏障：等这一步所有节点都结束，才统一按 Channel 规则合并写入。
    3. 路由：根据新 State 计算下一步的 frontier；然后存 Checkpoint。

为什么要屏障？并行节点互相看不到对方“这一步”的写入，结果与执行快慢无关，可复现。
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .observability import NullTracer, Tracer
from .workflow_checkpoint import Checkpoint, CheckpointStore, Status, utc_now
from .workflow_core import START, Graph, NodeContext, apply_writes


class Interrupt(Exception):  # 它表示“暂停”而不是错误，所以不叫 XxxError
    """节点抛出它表示“需要人工介入”。payload 是给人看的内容（问题、草稿……）。"""

    def __init__(self, payload: Any = None):
        super().__init__("workflow interrupted")
        self.payload = payload


@dataclass(slots=True)
class RunResult:
    thread_id: str
    workflow: str
    status: Status
    state: dict[str, Any]
    steps: int
    interrupt: dict[str, Any] | None = None
    error: str | None = None

    @property
    def output(self) -> Any:
        return self.state.get("output")


class WorkflowRunner:
    def __init__(
        self,
        graph: Graph,
        store: CheckpointStore,
        *,
        max_steps: int = 50,
        parallelism: int = 4,
        tracer: Tracer | None = None,
    ):
        if max_steps < 1 or parallelism < 1:
            raise ValueError("max_steps and parallelism must be positive")
        self.graph = graph.validate()
        self.store = store
        self.max_steps = max_steps
        self.parallelism = parallelism
        self.tracer = tracer or NullTracer()

    async def run(self, inputs: Mapping[str, Any], thread_id: str | None = None) -> RunResult:
        state = copy.deepcopy(dict(inputs))
        checkpoint = Checkpoint(
            thread_id=thread_id or uuid4().hex[:12],
            workflow=self.graph.name,
            status="running",
            step=0,
            state=state,
            frontier=self.graph.next_nodes([START], state),
        )
        self.tracer.emit("workflow.start", self.graph.name, thread_id=checkpoint.thread_id)
        await self._save(checkpoint)
        return await self._execute(checkpoint, resume={})

    async def resume(self, thread_id: str, value: Any = None) -> RunResult:
        """继续一个被中断（人工审批）或失败（重试）的工作流。"""
        checkpoint = await self.store.load(thread_id)
        if checkpoint is None:
            raise KeyError(f"checkpoint not found: {thread_id}")
        if checkpoint.workflow != self.graph.name:
            raise ValueError(f"checkpoint belongs to {checkpoint.workflow!r}")
        if checkpoint.status not in ("interrupted", "failed"):
            raise ValueError(f"cannot resume a {checkpoint.status} workflow")
        resume = {checkpoint.interrupt["node"]: value} if checkpoint.interrupt else {}
        checkpoint.status, checkpoint.interrupt, checkpoint.error = "running", None, None
        self.tracer.emit("workflow.start", self.graph.name, thread_id=thread_id)
        return await self._execute(checkpoint, resume)

    async def _execute(self, cp: Checkpoint, resume: dict[str, Any]) -> RunResult:
        semaphore = asyncio.Semaphore(self.parallelism)
        while cp.frontier:
            if cp.step >= self.max_steps:
                return await self._finish(
                    cp, "failed", error=f"exceeded max_steps={self.max_steps}"
                )
            self.tracer.emit("workflow.step", self.graph.name, step=cp.step + 1, nodes=cp.frontier)

            # ---- 1. 执行：中断 / 失败恢复时，已经完成的节点不再重跑
            todo = [name for name in cp.frontier if name not in cp.pending_writes]
            snapshot = MappingProxyType(copy.deepcopy(cp.state))  # 只读快照
            outcomes = await asyncio.gather(
                *(self._run_node(name, snapshot, cp, resume.get(name), semaphore) for name in todo),
                return_exceptions=True,
            )
            resume = {}  # 人工输入只用一次

            failed: list[str] = []
            interrupted: tuple[str, Any] | None = None
            for name, outcome in zip(todo, outcomes, strict=True):
                if isinstance(outcome, Interrupt):
                    interrupted = interrupted or (name, outcome.payload)
                elif isinstance(outcome, BaseException):
                    failed.append(f"{name}: {type(outcome).__name__}: {outcome}")
                else:
                    cp.pending_writes[name] = outcome
                    self.tracer.emit(
                        "workflow.node_done", self.graph.name, node=name, keys=list(outcome)
                    )
            if failed:
                return await self._finish(cp, "failed", error="; ".join(failed))
            if interrupted is not None:
                node, payload = interrupted
                cp.interrupt = {"node": node, "payload": payload}
                self.tracer.emit("workflow.interrupted", self.graph.name, node=node)
                return await self._finish(cp, "interrupted")

            # ---- 2. 屏障：全部完成后按 frontier 顺序合并，结果与谁先跑完无关
            # ---- 3. 路由：根据新 State 算出下一步
            try:
                cp.state = apply_writes(
                    self.graph.channels, cp.state, [cp.pending_writes[n] for n in cp.frontier]
                )
                cp.frontier = self.graph.next_nodes(cp.frontier, cp.state)
            except (ValueError, KeyError, TypeError) as error:
                return await self._finish(cp, "failed", error=f"{type(error).__name__}: {error}")
            cp.pending_writes = {}
            cp.step += 1
            await self._save(cp)  # 每一步都存档，崩溃后最多丢一步
        return await self._finish(cp, "completed")

    async def _run_node(
        self,
        name: str,
        state: Mapping[str, Any],
        cp: Checkpoint,
        resume_value: Any,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            ctx = NodeContext(thread_id=cp.thread_id, step=cp.step, resume=resume_value)
            result = await self.graph.nodes[name].run(state, ctx)
        if result is None:
            return {}
        if not isinstance(result, Mapping):
            raise TypeError(f"node must return a mapping, got {type(result).__name__}")
        return dict(result)

    async def _save(self, cp: Checkpoint) -> None:
        cp.updated_at = utc_now()
        await self.store.save(cp)

    async def _finish(self, cp: Checkpoint, status: Status, error: str | None = None) -> RunResult:
        cp.status, cp.error = status, error
        await self._save(cp)
        self.tracer.emit(
            "workflow.finish", self.graph.name, status=status, steps=cp.step, error=error
        )
        return RunResult(cp.thread_id, cp.workflow, status, cp.state, cp.step, cp.interrupt, error)
