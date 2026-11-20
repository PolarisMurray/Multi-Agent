"""任务 worker：从 Kafka 取 Gin 网关（gateway/）提交的任务，交给 WorkflowService 执行。

    bitguide.tasks    网关 → worker：{"id", "action": run|resume, "workflow", "input", "value"}
    bitguide.events   worker → 网关：{"id", "type": status|progress, ...}

- 两个 topic 都用任务 id 当 key：同一个任务的消息落在同一个分区，保证有序。
- 所有 worker 在同一个 consumer group 里，分区自动分给各个消费者；多开进程就是扩容。
- 执行完、事件也发出去之后才提交 offset：崩溃了消息会重投（at-least-once），
  所以执行前先查 Checkpoint，已经处理过的直接跳过（幂等）。
- worker 不碰 Redis：任务状态只由网关根据 events 写入，只有一个写入方。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .observability import Tracer
from .workflow import WorkflowService

if TYPE_CHECKING:
    from .application import Platform

TASK_TOPIC = "bitguide.tasks"
EVENT_TOPIC = "bitguide.events"
GROUP_ID = "bitguide-workers"

Publish = Callable[[dict[str, Any]], None]


class EventTracer:
    """把引擎的进度事件转成 progress 事件发出去，同时照常交给原来的 Tracer 打日志。"""

    def __init__(self, publish: Publish, inner: Tracer):
        self._publish = publish
        self._inner = inner

    def emit(self, kind: str, source: str, **data: Any) -> None:
        self._inner.emit(kind, source, **data)
        if kind in ("workflow.step", "workflow.node_done"):
            self._publish({"id": data.pop("thread_id"), "type": "progress", "kind": kind, **data})


async def already_done(service: WorkflowService, job: dict[str, Any]) -> bool:
    checkpoint = await service.get(job["id"])
    if checkpoint is None:
        return False
    if job["action"] == "run":
        return True  # 跑过了，结果已经在 Checkpoint 里
    return checkpoint.status not in ("interrupted", "failed")  # resume 只对这两种状态有意义


async def handle(service: WorkflowService, job: dict[str, Any], publish: Publish) -> None:
    if await already_done(service, job):
        print(f"[worker] skip {job['action']} {job['id']}（重复投递）", flush=True)
        return
    publish({"id": job["id"], "type": "status", "status": "running"})
    try:
        if job["action"] == "resume":
            result = await service.resume(job["id"], job.get("value"))
        else:
            result = await service.run(job["workflow"], {"input": job["input"]}, job["id"])
    except Exception as error:  # 工作流不存在之类：这个任务失败，worker 继续干活
        publish({"id": job["id"], "type": "status", "status": "failed", "error": str(error)})
        return
    # 节点报错不会抛到这里：runner 会返回 status="failed"，照常发回
    output, interrupt = result.output, result.interrupt
    publish(
        {
            "id": job["id"],
            "type": "status",
            "status": result.status,
            "steps": result.steps,
            "output": output if output is None or isinstance(output, str) else json.dumps(output),
            "interrupt": json.dumps(interrupt, ensure_ascii=False) if interrupt else None,
            "error": result.error,
        }
    )


async def serve(platform: Platform, brokers: str, concurrency: int = 3) -> None:
    # 延迟导入：只有跑 worker 时才需要安装 aiokafka（uv sync --extra kafka）
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    producer = AIOKafkaProducer(bootstrap_servers=brokers, acks="all")
    await producer.start()
    # 引擎的 emit 是同步的，发 Kafka 是异步的：先放进队列，由一个协程按顺序发出去
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def pump() -> None:
        while True:
            event = await events.get()
            value = json.dumps(event, ensure_ascii=False).encode()
            await producer.send_and_wait(EVENT_TOPIC, value, key=event["id"].encode())
            events.task_done()

    settings = platform.settings.workflow
    service = WorkflowService(
        platform.workflows.registry,
        platform.workflows.store,
        max_steps=settings.max_steps,
        parallelism=settings.parallelism,
        tracer=EventTracer(events.put_nowait, platform.tracer),
    )

    async def consume() -> None:
        consumer = AIOKafkaConsumer(
            TASK_TOPIC, bootstrap_servers=brokers, group_id=GROUP_ID, enable_auto_commit=False
        )
        await consumer.start()
        try:
            async for message in consumer:
                assert message.value is not None  # 网关发的消息一定有内容
                job = json.loads(message.value)
                print(f"[worker] {job['action']} {job['id']}", flush=True)
                await handle(service, job, events.put_nowait)
                await events.join()  # 事件都发出去了，才算处理完
                await consumer.commit()
        finally:
            await consumer.stop()

    # 一个进程里起 N 个消费者，同组共享分区；每个消费者按顺序处理，所以同一分区内有序
    print(f"[worker] 消费 {TASK_TOPIC}（group={GROUP_ID}），并发 {concurrency}", flush=True)
    sender = asyncio.create_task(pump())
    try:
        await asyncio.gather(*(consume() for _ in range(concurrency)))
    finally:
        sender.cancel()
        await producer.stop()
