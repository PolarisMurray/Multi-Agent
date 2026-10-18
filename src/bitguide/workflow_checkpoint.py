"""Checkpoint：每执行完一步就把“完整执行现场”存下来。

存了什么就能做什么：
- 进程崩溃 / 重启 → 读出 Checkpoint，从 frontier 继续跑（断点恢复）。
- 节点抛 Interrupt → 状态记为 interrupted，等人工输入后 resume（Human-in-the-loop）。
- 某个节点失败 → 状态记为 failed；已成功节点的写入在 pending_writes 里，重试只重跑失败的节点。

三种存储，同一个 CheckpointStore 协议：
    MemoryCheckpointStore  进程内 dict，测试用
    FileCheckpointStore    本地 JSON 文件，默认，零依赖也能跨进程恢复
    RedisCheckpointStore   多实例共享 + TTL 自动过期，生产用
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from .config import CheckpointSettings

Status = Literal["running", "interrupted", "completed", "failed"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Checkpoint:
    thread_id: str
    workflow: str
    status: Status
    step: int
    state: dict[str, Any]
    frontier: list[str]  # 当前这一步要执行的节点
    # 本步已经跑完的节点的写入；中断 / 失败后恢复时，这些节点不用重跑
    pending_writes: dict[str, dict[str, Any]] = field(default_factory=dict)
    interrupt: dict[str, Any] | None = None  # {"node": 节点名, "payload": 给人看的内容}
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(**copy.deepcopy(data))


class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def load(self, thread_id: str) -> Checkpoint | None: ...

    async def list(self, limit: int = 20) -> list[Checkpoint]: ...

    async def delete(self, thread_id: str) -> None: ...


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        # 存一份深拷贝：调用方之后再改 checkpoint，不会影响已经存下的快照
        self._data[checkpoint.thread_id] = copy.deepcopy(checkpoint.to_dict())

    async def load(self, thread_id: str) -> Checkpoint | None:
        data = self._data.get(thread_id)
        return Checkpoint.from_dict(data) if data else None

    async def list(self, limit: int = 20) -> list[Checkpoint]:
        items = sorted(self._data.values(), key=lambda d: d["updated_at"], reverse=True)
        return [Checkpoint.from_dict(item) for item in items[:limit]]

    async def delete(self, thread_id: str) -> None:
        self._data.pop(thread_id, None)


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class FileCheckpointStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        # thread_id 会拼进文件路径，必须校验，防止 "../../etc/passwd" 这种路径穿越
        if not _SAFE_ID.match(thread_id) or thread_id.startswith("."):
            raise ValueError(f"invalid thread id: {thread_id!r}")
        return self.directory / f"{thread_id}.json"

    async def save(self, checkpoint: Checkpoint) -> None:
        path = self._path(checkpoint.thread_id)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2), "utf-8")
        # 先写临时文件再原子替换：写到一半崩溃也不会留下半个 JSON
        os.replace(temp, path)

    async def load(self, thread_id: str) -> Checkpoint | None:
        path = self._path(thread_id)
        if not path.exists():
            return None
        return Checkpoint.from_dict(json.loads(path.read_text("utf-8")))

    async def list(self, limit: int = 20) -> list[Checkpoint]:
        items = [json.loads(p.read_text("utf-8")) for p in self.directory.glob("*.json")]
        items.sort(key=lambda d: d["updated_at"], reverse=True)
        return [Checkpoint.from_dict(item) for item in items[:limit]]

    async def delete(self, thread_id: str) -> None:
        self._path(thread_id).unlink(missing_ok=True)


class RedisCheckpointStore:
    """每个 thread 一个 JSON 字符串（带 TTL）+ 一个按更新时间排序的 ZSET 索引。"""

    def __init__(
        self, client: Redis, prefix: str = "bitguide:checkpoint:", ttl_seconds: int = 604800
    ):
        self._redis = client
        self._prefix = prefix
        self._index = prefix + "index"
        self._ttl = ttl_seconds

    async def save(self, checkpoint: Checkpoint) -> None:
        payload = json.dumps(checkpoint.to_dict(), ensure_ascii=False)
        # MULTI/EXEC 事务：快照和索引要么一起更新，要么都不更新
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(self._prefix + checkpoint.thread_id, payload, ex=self._ttl)
            pipe.zadd(self._index, {checkpoint.thread_id: time.time()})
            await pipe.execute()

    async def load(self, thread_id: str) -> Checkpoint | None:
        raw = await self._redis.get(self._prefix + thread_id)
        return Checkpoint.from_dict(json.loads(raw)) if raw else None

    async def list(self, limit: int = 20) -> list[Checkpoint]:
        ids = await self._redis.zrevrange(self._index, 0, limit - 1)
        result = []
        for raw_id in ids:
            thread_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            checkpoint = await self.load(thread_id)
            if checkpoint is None:  # 快照已经 TTL 过期，顺手清理索引
                await self._redis.zrem(self._index, thread_id)
            else:
                result.append(checkpoint)
        return result

    async def delete(self, thread_id: str) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(self._prefix + thread_id)
            pipe.zrem(self._index, thread_id)
            await pipe.execute()

    async def close(self) -> None:
        await self._redis.aclose()


def create_checkpoint_store(settings: CheckpointSettings, base_dir: Path) -> CheckpointStore:
    if settings.backend == "memory":
        return MemoryCheckpointStore()
    if settings.backend == "redis":
        from redis.asyncio import Redis  # 延迟导入：只有用 Redis 时才需要安装 redis 包

        return RedisCheckpointStore(
            Redis.from_url(settings.redis_url), settings.prefix, settings.ttl_seconds
        )
    directory = (
        settings.directory if settings.directory.is_absolute() else base_dir / settings.directory
    )
    return FileCheckpointStore(directory)
