"""AgentFactory：配置世界 → 运行时对象 的唯一边界。

配置（YAML / etcd）里只有“声明”：用哪个模型、哪些工具、哪个知识库、哪些 Skill。
Factory 负责把声明变成真正的对象，并把这些资源的所有权交给 Agent（agent.close() 时释放）。
"""

from __future__ import annotations

import asyncio
import sys

from .agent import Agent
from .config import AgentSettings, PlatformSettings
from .contracts import Closeable, LLMClient
from .llm import OfflineLLM, OpenAICompatibleClient
from .mcp import connect_stdio_server
from .memory import ConversationMemory
from .observability import NullTracer, Tracer
from .rag import (
    Embedder,
    HashEmbedder,
    InMemoryVectorStore,
    MilvusVectorStore,
    OllamaEmbedder,
    RagRetriever,
    TextSplitter,
    VectorStore,
    load_documents,
)
from .skills import SkillManager
from .tools import create_builtin_tools


class AgentFactory:
    def __init__(self, settings: PlatformSettings, tracer: Tracer | None = None):
        self.settings = settings
        self.tracer = tracer or NullTracer()

    async def build(self, spec: AgentSettings) -> Agent:
        self.settings.check_agent(spec)
        resources: list[Closeable] = []
        try:
            llm = self.build_llm(spec.model_ref)
            if isinstance(llm, Closeable):  # OfflineLLM 没有连接要关，真实客户端有
                resources.append(llm)

            tools = list(create_builtin_tools(spec.tools))
            for server_name in spec.mcp_servers:
                tools.extend(await self._connect_mcp(spec.id, server_name, resources))

            retriever = await self._build_retriever(spec.id, spec.knowledge_base)
            if retriever is not None:
                resources.append(retriever)

            skills = None
            if spec.skills:
                skills = SkillManager(
                    self.settings.resolve(self.settings.skills.directory), spec.skills
                )

            return Agent(
                spec.id,
                spec.system_prompt,
                llm,
                tools,
                retriever,
                spec.max_iterations,
                ConversationMemory(spec.max_history_length),
                name=spec.name,
                description=spec.description,
                skills=skills,
                tracer=self.tracer,
                resources=resources,
            )
        except Exception:
            # 装配到一半失败：把已经创建的资源（子进程、连接）全部关掉再抛出
            await asyncio.gather(*(r.close() for r in resources), return_exceptions=True)
            raise

    def build_llm(self, model_ref: str) -> LLMClient:
        model = self.settings.models[model_ref]
        if model.provider == "offline":
            return OfflineLLM()
        return OpenAICompatibleClient(
            base_url=model.base_url,
            api_key=model.resolved_api_key(),
            model=model.model_name,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            timeout=model.timeout_seconds,
        )

    async def _connect_mcp(self, agent_id: str, server_name: str, resources: list[Closeable]):
        server = self.settings.mcp_servers[server_name]
        # "python" 指向当前解释器，保证子进程和主进程用同一个虚拟环境
        command = sys.executable if server.command in ("python", "python3") else server.command
        try:
            client, tools = await connect_stdio_server(
                server_name,
                [command, *server.args],
                cwd=self.settings.base_dir,
                env=server.env,
                timeout_seconds=server.timeout_seconds,
            )
        except Exception as error:
            # 一个 MCP Server 起不来，不应该让整个 Agent 不可用：降级为“没有这些工具”
            self.tracer.emit(
                "agent.degraded", agent_id, component=f"MCP {server_name}", error=str(error)
            )
            return []
        resources.append(client)
        return tools

    def _build_embedder(self) -> Embedder:
        embedding = self.settings.rag.embedding
        if embedding.provider == "ollama":
            return OllamaEmbedder(
                embedding.base_url, embedding.model_name, embedding.timeout_seconds
            )
        return HashEmbedder(embedding.dimension)

    def _build_store(self, knowledge_base: str) -> VectorStore:
        store = self.settings.rag.vector_store
        if store.type == "milvus":
            uri = store.uri
            if not uri.startswith(("http://", "https://")):  # 本地文件 → Milvus Lite
                path = self.settings.resolve(uri)
                path.parent.mkdir(parents=True, exist_ok=True)
                uri = str(path)
            collection = f"{store.collection_prefix}{knowledge_base}"
            return MilvusVectorStore(uri, collection, self.settings.rag.embedding.dimension)
        return InMemoryVectorStore()

    async def _build_retriever(
        self, agent_id: str, knowledge_base: str | None
    ) -> RagRetriever | None:
        rag = self.settings.rag
        if not knowledge_base or not rag.enabled:
            return None
        retriever: RagRetriever | None = None
        try:
            store = self._build_store(knowledge_base)
            retriever = RagRetriever(
                self._build_embedder(),
                store,
                top_k=rag.retrieval.top_k,
                threshold=rag.retrieval.score_threshold,
                fail_open=rag.fail_open,
            )
            # 持久化的向量库（Milvus）已经有数据就不再重复向量化
            # TODO(改写): 文档更新后需要增量同步，可以比较文件 hash 决定是否重建
            if await store.count() == 0:
                documents = load_documents(
                    self.settings.resolve(self.settings.knowledge_bases[knowledge_base])
                )
                splitter = TextSplitter(rag.splitter.max_segment_size, rag.splitter.overlap)
                await retriever.ingest(documents, splitter)
            return retriever
        except Exception as error:
            if retriever is not None:
                await retriever.close()
            if not rag.fail_open:
                raise
            self.tracer.emit("agent.degraded", agent_id, component="RAG", error=str(error))
            return None
