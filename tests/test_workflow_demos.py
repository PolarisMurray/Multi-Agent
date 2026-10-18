"""端到端：用项目真实配置（离线模式）跑通全部 8 个演示工作流。"""

import pytest

from bitguide.application import Platform


@pytest.fixture
async def platform(offline_settings):
    platform = await Platform.create(offline_settings, trace=False)
    yield platform
    await platform.close()


async def test_all_eight_demos_are_registered(platform):
    assert platform.workflows.registry.names() == [
        "expert_review",
        "human_approval",
        "parallel_experts",
        "pipeline",
        "review_loop",
        "router",
        "supervisor",
        "tool_pipeline",
    ]


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("pipeline", "3 岁孩子发脾气怎么办"),
        ("router", "python 代码有 bug"),
        ("parallel_experts", "孩子想学编程"),
        ("review_loop", "睡前故事"),
        ("supervisor", "做一个儿童学习计划 App"),
        ("tool_pipeline", "(3+5)*12"),
    ],
)
async def test_demo_completes(platform, name, text):
    result = await platform.workflows.run(name, {"input": text})
    assert result.status == "completed", result.error
    assert result.output


async def test_router_picks_the_tech_branch(platform):
    result = await platform.workflows.run("router", {"input": "python 代码有 bug"})
    assert result.state["route"] == "tech"
    assert "技术助手" in result.output


async def test_parallel_experts_collects_both_opinions(platform):
    result = await platform.workflows.run("parallel_experts", {"input": "孩子想学编程"})
    labels = [opinion[:6] for opinion in result.state["opinions"]]
    assert labels == ["【亲子顾问】", "【技术助手】"]


async def test_review_loop_stops_after_max_rounds(platform):
    result = await platform.workflows.run("review_loop", {"input": "睡前故事"})
    assert result.state["round"] == 3


async def test_supervisor_dispatches_each_worker(platform):
    result = await platform.workflows.run("supervisor", {"input": "做一个 App"})
    assert result.state["dispatched"] == ["parenting", "tech"]
    assert len(result.state["notes"]) == 2


async def test_tool_pipeline_uses_the_real_calculator(platform):
    result = await platform.workflows.run("tool_pipeline", {"input": "(3+5)*12"})
    assert result.state["calc"] == "(3+5)*12 = 96"


async def test_human_approval_reject_then_approve(platform):
    service = platform.workflows
    paused = await service.run("human_approval", {"input": "周五团建"}, "hitl")
    assert paused.status == "interrupted"

    again = await service.resume("hitl", {"approved": False, "comment": "写上集合时间"})
    assert again.status == "interrupted"  # 修改后再次等待审批
    assert again.state["comment"] == "写上集合时间"

    done = await service.resume("hitl", {"approved": True})
    assert done.status == "completed"
    assert done.output.startswith("✅ 已发布")


async def test_expert_review_milestone(platform):
    service = platform.workflows
    paused = await service.run("expert_review", {"input": "孩子学编程"}, "milestone")
    assert paused.status == "interrupted" and paused.interrupt
    assert paused.interrupt["node"] == "approval"
    assert len(paused.state["opinions"]) == 2
    done = await service.resume("milestone", {"approved": True})
    assert done.status == "completed" and done.output == done.state["draft"]
    assert [c.thread_id for c in await service.history()] == ["milestone"]
