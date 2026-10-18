"""应用装配：把 配置 → Factory → Registry → 工作流 串起来。

CLI、测试、以后的 HTTP API 都只和 Platform 打交道，不直接 new 各种组件。
启动顺序：
    1. 读 YAML（可选：用 etcd 里的 Agent 定义覆盖 YAML 里的同名定义）
    2. AgentFactory 逐个装配 Agent，注册进 AgentRegistry
    3. 注册 8 个演示工作流，准备 Checkpoint 存储
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .config import AgentSettings, PlatformSettings
from .contracts import Closeable, LLMClient
from .factory import AgentFactory
from .infrastructure import EtcdClient
from .llm import OfflineLLM
from .models import AgentRequest, AgentResponse
from .observability import ConsoleTracer, JsonLogTracer, NullTracer, Tracer
from .registry import AgentRegistry
from .workflow import WorkflowService
from .workflow_checkpoint import create_checkpoint_store
from .workflow_demos import build_demo_workflows
from .workflow_registry import WorkflowRegistry

DEFAULT_CONFIG = Path("config/platform.yaml")


def build_tracer(settings: PlatformSettings, trace: bool | None = None) -> Tracer:
    enabled = settings.observability.trace if trace is None else trace
    if not enabled:
        return NullTracer()
    if settings.observability.format == "json":
        return JsonLogTracer()
    return ConsoleTracer(max_length=settings.observability.max_content_length)


async def load_agents_from_etcd(etcd: EtcdClient, prefix: str) -> dict[str, AgentSettings]:
    """etcd 里每个 key 是 <prefix><agent_id>，value 是 Agent 定义的 JSON。"""
    raw = await etcd.get_prefix(prefix)
    specs = (AgentSettings.model_validate(json.loads(value)) for value in raw.values())
    return {spec.id: spec for spec in specs}


async def seed_agents_to_etcd(etcd: EtcdClient, prefix: str, specs: list[AgentSettings]) -> None:
    for spec in specs:
        value = spec.model_dump_json(by_alias=True, exclude_none=True)
        await etcd.put(f"{prefix}{spec.id}", value)


class Platform:
    def __init__(
        self,
        settings: PlatformSettings,
        tracer: Tracer,
        factory: AgentFactory,
        agents: AgentRegistry,
        workflows: WorkflowService,
        resources: list[Closeable],
    ):
        self.settings = settings
        self.tracer = tracer
        self.factory = factory
        self.agents = agents
        self.workflows = workflows
        self._resources = resources

    @classmethod
    async def create(
        cls,
        config: str | Path | PlatformSettings = DEFAULT_CONFIG,
        *,
        offline: bool = False,
        trace: bool | None = None,
        use_etcd: bool | None = None,
        only_agents: list[str] | None = None,
    ) -> Platform:
        # 传路径就读文件；测试里也可以直接传一个改好的 PlatformSettings
        settings = config if isinstance(config, PlatformSettings) else PlatformSettings.load(config)
        if offline:
            settings = settings.offline()
        tracer = build_tracer(settings, trace)
        factory = AgentFactory(settings, tracer)
        resources: list[Closeable] = []

        specs = dict(settings.agents)
        if settings.etcd.enabled if use_etcd is None else use_etcd:
            etcd = EtcdClient(settings.etcd.endpoint, settings.etcd.timeout_seconds)
            try:
                remote = await load_agents_from_etcd(etcd, settings.etcd.prefix)
            finally:
                await etcd.close()
            for spec in remote.values():
                settings.check_agent(spec)
            specs.update(remote)  # 同名时 etcd 覆盖 YAML：配置中心是唯一事实来源
            # TODO(改写): 第 15 课——启动后 etcd.watch(prefix)，收到 PUT 就
            #   factory.build(新定义) → agents.replace(新 Agent)；收到 DELETE 就 agents.remove(id)

        agents = AgentRegistry()
        selected = [s for s in specs.values() if only_agents is None or s.id in only_agents]
        # 各 Agent 的装配互不依赖（各自起 MCP 子进程、各自建索引），可以并发
        built = await asyncio.gather(
            *(factory.build(spec) for spec in selected), return_exceptions=True
        )
        failures = [b for b in built if isinstance(b, BaseException)]
        for agent in built:
            if not isinstance(agent, BaseException):
                await agents.register(agent)
        if failures:
            await agents.close()
            raise failures[0]

        # 主管节点用第一个模型做调度决策；离线模型不会做决策，主管退回“按顺序派发”
        supervisor_llm: LLMClient | None = None
        if settings.models:
            llm = factory.build_llm(next(iter(settings.models)))
            if not isinstance(llm, OfflineLLM):
                supervisor_llm = llm
            if isinstance(llm, Closeable):
                resources.append(llm)
        registry = WorkflowRegistry()
        for graph in build_demo_workflows(agents, supervisor_llm):
            registry.register(graph)
        store = create_checkpoint_store(settings.workflow.checkpoint, settings.base_dir)
        if isinstance(store, Closeable):
            resources.append(store)
        workflows = WorkflowService(
            registry,
            store,
            max_steps=settings.workflow.max_steps,
            parallelism=settings.workflow.parallelism,
            tracer=tracer,
        )
        return cls(settings, tracer, factory, agents, workflows, resources)

    async def chat(self, agent_id: str, message: str, session_id: str = "cli") -> AgentResponse:
        async with self.agents.acquire(agent_id) as agent:
            return await agent.run(AgentRequest(message, session_id))

    async def close(self) -> None:
        await self.agents.close()
        await asyncio.gather(*(r.close() for r in self._resources), return_exceptions=True)

    async def __aenter__(self) -> Platform:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
