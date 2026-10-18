"""工作流的对外门面：CLI / 测试 / 以后的 HTTP API 都只调用 WorkflowService。

关键点：resume 时不依赖“内存里还留着哪个 runner”，而是读 Checkpoint 里的 workflow 名字，
重新找到 Graph、新建 runner。所以即使进程重启过，也能从断点继续。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .observability import Tracer
from .workflow_checkpoint import Checkpoint, CheckpointStore
from .workflow_registry import WorkflowRegistry
from .workflow_runtime import RunResult, WorkflowRunner


class WorkflowService:
    def __init__(
        self,
        registry: WorkflowRegistry,
        store: CheckpointStore,
        *,
        max_steps: int = 50,
        parallelism: int = 4,
        tracer: Tracer | None = None,
    ):
        self.registry = registry
        self.store = store
        self._max_steps = max_steps
        self._parallelism = parallelism
        self._tracer = tracer

    def _runner(self, name: str) -> WorkflowRunner:
        return WorkflowRunner(
            self.registry.get(name),
            self.store,
            max_steps=self._max_steps,
            parallelism=self._parallelism,
            tracer=self._tracer,
        )

    async def run(
        self, name: str, inputs: Mapping[str, Any], thread_id: str | None = None
    ) -> RunResult:
        return await self._runner(name).run(inputs, thread_id)

    async def resume(self, thread_id: str, value: Any = None) -> RunResult:
        checkpoint = await self.store.load(thread_id)
        if checkpoint is None:
            raise KeyError(f"checkpoint not found: {thread_id}")
        return await self._runner(checkpoint.workflow).resume(thread_id, value)

    async def get(self, thread_id: str) -> Checkpoint | None:
        return await self.store.load(thread_id)

    async def history(self, limit: int = 20) -> list[Checkpoint]:
        return await self.store.list(limit)
