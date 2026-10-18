import fakeredis
import pytest

from bitguide.config import CheckpointSettings
from bitguide.workflow_checkpoint import (
    Checkpoint,
    FileCheckpointStore,
    MemoryCheckpointStore,
    RedisCheckpointStore,
    create_checkpoint_store,
)


def checkpoint(thread_id: str, updated_at: str) -> Checkpoint:
    return Checkpoint(
        thread_id=thread_id,
        workflow="wf",
        status="interrupted",
        step=2,
        state={"input": "你好", "notes": ["a"]},
        frontier=["approval"],
        pending_writes={"worker": {"x": 1}},
        interrupt={"node": "approval", "payload": {"question": "?"}},
        updated_at=updated_at,
    )


@pytest.fixture(params=["memory", "file", "redis"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryCheckpointStore()
    if request.param == "file":
        return FileCheckpointStore(tmp_path)
    return RedisCheckpointStore(fakeredis.FakeAsyncRedis(), ttl_seconds=60)


async def test_round_trip_list_and_delete(store):
    first, second = checkpoint("t1", "2026-01-01"), checkpoint("t2", "2026-01-02")
    await store.save(first)
    await store.save(second)
    assert await store.load("t1") == first
    assert await store.load("missing") is None
    assert [c.thread_id for c in await store.list()] == ["t2", "t1"]  # 最近更新的在前
    await store.delete("t1")
    assert await store.load("t1") is None


async def test_saved_snapshot_is_isolated_from_later_mutation():
    store = MemoryCheckpointStore()
    cp = checkpoint("t", "2026-01-01")
    await store.save(cp)
    cp.state["input"] = "changed"
    loaded = await store.load("t")
    assert loaded is not None and loaded.state["input"] == "你好"


async def test_file_store_rejects_path_traversal(tmp_path):
    store = FileCheckpointStore(tmp_path)
    for bad in ("../evil", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            await store.load(bad)


async def test_redis_store_sets_ttl_and_cleans_stale_index():
    redis = fakeredis.FakeAsyncRedis()
    store = RedisCheckpointStore(redis, prefix="cp:", ttl_seconds=60)
    await store.save(checkpoint("t", "2026-01-01"))
    assert 0 < await redis.ttl("cp:t") <= 60
    await redis.delete("cp:t")  # 模拟 TTL 过期
    assert await store.list() == []
    assert await redis.zcard("cp:index") == 0
    await store.close()


def test_store_factory(tmp_path):
    memory = create_checkpoint_store(CheckpointSettings(backend="memory"), tmp_path)
    assert isinstance(memory, MemoryCheckpointStore)
    file_store = create_checkpoint_store(CheckpointSettings(directory=tmp_path / "cp"), tmp_path)
    assert isinstance(file_store, FileCheckpointStore) and (tmp_path / "cp").is_dir()
    redis_store = create_checkpoint_store(CheckpointSettings(backend="redis"), tmp_path)
    assert isinstance(redis_store, RedisCheckpointStore)
