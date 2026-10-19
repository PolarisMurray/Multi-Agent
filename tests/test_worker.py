"""worker：模拟 Gin 网关往 Redis 里写任务，检查 worker 执行后写回的状态。"""

import asyncio
import json

import fakeredis
import pytest

from bitguide.application import Platform
from bitguide.worker import QUEUE_KEY, TASK_PREFIX, consume, handle


@pytest.fixture
async def platform(offline_settings):
    platform = await Platform.create(offline_settings, trace=False)
    yield platform
    await platform.close()


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


async def test_run_then_resume_after_approval(platform, redis):
    job = {"id": "t1", "action": "run", "workflow": "human_approval", "input": "周五团建"}
    await handle(platform.workflows, redis, job)
    task = await redis.hgetall(TASK_PREFIX + "t1")
    assert task["status"] == "interrupted"
    assert json.loads(task["interrupt"])["node"] and task["output"] == ""

    job = {"id": "t1", "action": "resume", "value": {"approved": True, "comment": ""}}
    await handle(platform.workflows, redis, job)
    task = await redis.hgetall(TASK_PREFIX + "t1")
    assert task["status"] == "completed" and "已发布" in task["output"]
    assert task["interrupt"] == ""  # 旧的中断信息被清掉


async def test_unknown_workflow_marks_task_failed(platform, redis):
    job = {"id": "t2", "action": "run", "workflow": "nope", "input": "x"}
    await handle(platform.workflows, redis, job)
    task = await redis.hgetall(TASK_PREFIX + "t2")
    assert task["status"] == "failed" and "nope" in task["error"]


async def test_consumer_takes_jobs_from_queue(platform, redis):
    job = {"id": "t3", "action": "run", "workflow": "pipeline", "input": "孩子想学编程"}
    await redis.lpush(QUEUE_KEY, json.dumps(job))
    consumer = asyncio.create_task(consume(platform.workflows, redis))
    try:
        for _ in range(100):
            if await redis.hget(TASK_PREFIX + "t3", "status") == "completed":
                break
            await asyncio.sleep(0.02)
        assert await redis.hget(TASK_PREFIX + "t3", "status") == "completed"
        assert await redis.llen(QUEUE_KEY) == 0
    finally:
        consumer.cancel()
