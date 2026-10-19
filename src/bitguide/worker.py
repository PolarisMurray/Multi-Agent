"""任务 worker：从 Redis 队列取 Gin 网关（gateway/main.go）提交的任务，交给 WorkflowService 执行。

和网关约定的 Redis 结构：
    bitguide:tasks        List，网关 LPUSH 任务 JSON，worker BRPOP 取走
    bitguide:task:<id>    Hash，任务状态：queued → running → completed / interrupted / failed
任务 id 直接当工作流的 thread id，所以 resume 能找到同一个 Checkpoint。

一个 worker 进程里起 N 个消费者协程，最多同时跑 N 个任务；要更多吞吐就多开几个 worker 进程，
它们从同一个队列抢任务，网关不用改。
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from .workflow import WorkflowService
from .workflow_checkpoint import utc_now

if TYPE_CHECKING:
    from redis.asyncio import Redis

QUEUE_KEY = "bitguide:tasks"
TASK_PREFIX = "bitguide:task:"


async def handle(service: WorkflowService, redis: Redis, job: dict[str, Any]) -> None:
    key = TASK_PREFIX + job["id"]
    await redis.hset(key, mapping={"status": "running", "updated_at": utc_now()})
    try:
        if job["action"] == "resume":
            result = await service.resume(job["id"], job.get("value"))
        else:
            result = await service.run(job["workflow"], {"input": job["input"]}, job["id"])
    except Exception as error:  # 工作流不存在之类：这个任务失败，worker 继续干活
        await redis.hset(
            key, mapping={"status": "failed", "error": str(error), "updated_at": utc_now()}
        )
        return
    # 节点报错不会抛到这里：runner 会返回 status="failed"，照常写回。
    # 每个字段都写：resume 之后旧的 interrupt / error 会被清空
    output, interrupt = result.output, result.interrupt
    if output is not None and not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    await redis.hset(
        key,
        mapping={
            "status": result.status,
            "steps": result.steps,
            "output": output or "",
            "interrupt": json.dumps(interrupt, ensure_ascii=False) if interrupt else "",
            "error": result.error or "",
            "updated_at": utc_now(),
        },
    )


async def consume(service: WorkflowService, redis: Redis) -> None:
    while True:
        # 阻塞等待，队列空时不占 CPU。超时要比客户端读超时（redis-py 默认 5 秒）短，空了就再等一轮
        item = await redis.brpop([QUEUE_KEY], timeout=1)
        if item is None:
            continue
        job = json.loads(item[1])
        print(f"[worker] {job['action']} {job['id']}", flush=True)
        await handle(service, redis, job)


async def serve(service: WorkflowService, redis_url: str, concurrency: int = 4) -> None:
    from redis.asyncio import Redis  # 延迟导入：只有跑 worker 时才需要安装 redis 包

    redis = Redis.from_url(redis_url, decode_responses=True)
    print(f"[worker] 监听 {QUEUE_KEY}，并发 {concurrency}", flush=True)
    try:
        await asyncio.gather(*(consume(service, redis) for _ in range(concurrency)))
    finally:
        await redis.aclose()
