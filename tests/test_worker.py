"""worker：不连 Kafka，publish 换成往列表里追加，检查发出的事件和幂等跳过。"""

import json

import pytest

from bitguide.application import Platform
from bitguide.observability import NullTracer
from bitguide.worker import EventTracer, handle
from bitguide.workflow import WorkflowService


@pytest.fixture
async def platform(offline_settings):
    platform = await Platform.create(offline_settings, trace=False)
    yield platform
    await platform.close()


@pytest.fixture
def events():
    return []


@pytest.fixture
def service(platform, events):
    tracer = EventTracer(events.append, NullTracer())
    return WorkflowService(platform.workflows.registry, platform.workflows.store, tracer=tracer)


def statuses(events):
    return [e["status"] for e in events if e["type"] == "status"]


async def test_run_then_resume_after_approval(service, events):
    await handle(
        service,
        {"id": "t1", "action": "run", "workflow": "human_approval", "input": "周五团建"},
        events.append,
    )
    assert statuses(events) == ["running", "interrupted"]
    assert json.loads(events[-1]["interrupt"])["node"]
    assert any(e["type"] == "progress" and e["id"] == "t1" for e in events)

    events.clear()
    value = {"approved": True, "comment": ""}
    await handle(service, {"id": "t1", "action": "resume", "value": value}, events.append)
    assert statuses(events) == ["running", "completed"]
    assert "已发布" in events[-1]["output"] and events[-1]["interrupt"] is None


async def test_redelivered_messages_are_skipped(service, events):
    job = {"id": "t2", "action": "run", "workflow": "pipeline", "input": "孩子想学编程"}
    await handle(service, job, events.append)
    assert statuses(events) == ["running", "completed"]

    events.clear()
    await handle(service, job, events.append)  # 同一条消息又来了一次
    await handle(service, {"id": "t2", "action": "resume"}, events.append)  # 已完成的不能再 resume
    assert events == []


async def test_unknown_workflow_reports_failed(service, events):
    await handle(
        service, {"id": "t3", "action": "run", "workflow": "nope", "input": "x"}, events.append
    )
    assert statuses(events) == ["running", "failed"] and "nope" in events[-1]["error"]
