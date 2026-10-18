"""强类型配置：YAML → Pydantic 模型。

设计要点：
- YAML 用 camelCase（和原 Java 版保持一致），Python 里用 snake_case，靠 alias_generator 自动对应。
- extra="forbid"：YAML 里拼错一个键，启动时就报错，而不是悄悄用默认值。
- 引用校验：Agent 引用的模型 / 知识库 / MCP Server 必须存在，在加载时统一检查。
- 相对路径一律相对 baseDir（默认是配置文件所在目录的上一级，也就是项目根）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator
from pydantic.alias_generators import to_camel


class _Settings(BaseModel):
    # alias_generator：Python 字段 max_segment_size 自动对应 YAML 里的 maxSegmentSize
    model_config = ConfigDict(
        alias_generator=to_camel, validate_by_name=True, validate_by_alias=True, extra="forbid"
    )


class ModelSettings(_Settings):
    provider: Literal["ollama", "openai", "offline"] = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    # 推荐写法：apiKeyEnv: OPENAI_API_KEY，密钥放环境变量，不进 Git
    api_key_env: str | None = None
    model_name: str = "qwen2.5"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: PositiveInt = 2000
    timeout_seconds: float = Field(default=60, gt=0)

    def resolved_api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return self.api_key


class EmbeddingSettings(_Settings):
    # hash：本地哈希向量，离线可用但不懂语义；ollama：真正的 Embedding 模型
    provider: Literal["hash", "ollama"] = "hash"
    base_url: str = "http://localhost:11434"
    model_name: str = "nomic-embed-text"
    dimension: PositiveInt = 256
    timeout_seconds: float = Field(default=30, gt=0)


class VectorStoreSettings(_Settings):
    type: Literal["memory", "milvus"] = "memory"
    # milvus 时：本地文件（Milvus Lite）或 http://localhost:19530
    uri: str = "data/milvus.db"
    collection_prefix: str = "bitguide_"


class SplitterSettings(_Settings):
    max_segment_size: PositiveInt = 300
    overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> SplitterSettings:
        if self.overlap >= self.max_segment_size:
            raise ValueError("overlap must be smaller than maxSegmentSize")
        return self


class RetrievalSettings(_Settings):
    top_k: PositiveInt = 3
    score_threshold: float = Field(default=0.1, ge=-1, le=1)


class RagSettings(_Settings):
    enabled: bool = True
    # fail-open：检索出错时降级为“无资料”继续回答，而不是让整个请求失败
    fail_open: bool = True
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    splitter: SplitterSettings = Field(default_factory=SplitterSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


class McpServerSettings(_Settings):
    # TODO(改写): 目前只支持 stdio 传输；HTTP（Streamable HTTP）传输留给你实现
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30, gt=0)


class SkillsSettings(_Settings):
    directory: Path = Path("skills")


class ObservabilitySettings(_Settings):
    trace: bool = True
    format: Literal["console", "json"] = "console"
    max_content_length: PositiveInt = 300


class AgentSettings(_Settings):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    model_ref: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    knowledge_base: str | None = None
    max_iterations: PositiveInt = 8
    max_history_length: PositiveInt = 50


class EtcdSettings(_Settings):
    enabled: bool = False
    endpoint: str = "http://localhost:2379"
    prefix: str = "/bitguide/agents/"
    timeout_seconds: float = Field(default=5, gt=0)

    @field_validator("prefix")
    @classmethod
    def _ends_with_slash(cls, value: str) -> str:
        return value if value.endswith("/") else value + "/"


class CheckpointSettings(_Settings):
    # file：默认，进程重启后仍能恢复；redis：多实例共享；memory：只在测试里用
    backend: Literal["memory", "file", "redis"] = "file"
    directory: Path = Path(".bitguide/checkpoints")
    redis_url: str = "redis://localhost:6379/0"
    prefix: str = "bitguide:checkpoint:"
    ttl_seconds: PositiveInt = 7 * 24 * 3600


class WorkflowSettings(_Settings):
    max_steps: PositiveInt = 50
    parallelism: PositiveInt = 4
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)


class PlatformSettings(_Settings):
    base_dir: Path = Path("..")
    models: dict[str, ModelSettings]
    agents: dict[str, AgentSettings] = Field(default_factory=dict)
    knowledge_bases: dict[str, Path] = Field(default_factory=dict)
    mcp_servers: dict[str, McpServerSettings] = Field(default_factory=dict)
    rag: RagSettings = Field(default_factory=RagSettings)
    skills: SkillsSettings = Field(default_factory=SkillsSettings)
    etcd: EtcdSettings = Field(default_factory=EtcdSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def _check_references(self) -> PlatformSettings:
        for key, agent in self.agents.items():
            self.check_agent(agent, key)
        return self

    def check_agent(self, agent: AgentSettings, key: str | None = None) -> None:
        """单独暴露出来：从 etcd 读到的 Agent 定义也要过同一套校验。"""
        if key is not None and key != agent.id:
            raise ValueError(f"agents key '{key}' must equal id '{agent.id}'")
        if agent.model_ref not in self.models:
            raise ValueError(f"Agent '{agent.id}' references unknown model '{agent.model_ref}'")
        if agent.knowledge_base and agent.knowledge_base not in self.knowledge_bases:
            raise ValueError(
                f"Agent '{agent.id}' references unknown knowledge base '{agent.knowledge_base}'"
            )
        for server in agent.mcp_servers:
            if server not in self.mcp_servers:
                raise ValueError(f"Agent '{agent.id}' references unknown MCP server '{server}'")

    @classmethod
    def load(cls, path: str | Path) -> PlatformSettings:
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"configuration root must be a mapping: {source}")
        settings = cls.model_validate(raw)
        settings.base_dir = (source.parent / settings.base_dir).resolve()
        return settings

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.base_dir / candidate

    def offline(self) -> PlatformSettings:
        """返回一份“全离线”配置：模型换成 OfflineLLM，向量换成本地哈希 + 内存库。"""
        rag = self.rag.model_copy(
            update={
                "embedding": EmbeddingSettings(provider="hash"),
                "vector_store": VectorStoreSettings(type="memory"),
                "retrieval": self.rag.retrieval.model_copy(update={"score_threshold": 0.1}),
            }
        )
        models = {k: m.model_copy(update={"provider": "offline"}) for k, m in self.models.items()}
        return self.model_copy(update={"models": models, "rag": rag})
