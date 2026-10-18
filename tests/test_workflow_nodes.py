import pytest
from conftest import ScriptedLLM

from bitguide.agent import Agent
from bitguide.models import ModelTurn
from bitguide.registry import AgentRegistry
from bitguide.tools import CalculatorTool
from bitguide.workflow_core import NodeContext
from bitguide.workflow_nodes import (
    FINISH,
    AgentNode,
    FunctionNode,
    HumanApprovalNode,
    LoopNode,
    RouterNode,
    SupervisorNode,
    ToolNode,
    render,
)
from bitguide.workflow_runtime import Interrupt

CTX = NodeContext(thread_id="t", step=0)


async def run(node, state, ctx: NodeContext = CTX) -> dict:
    """调用节点并断言它返回了写入（而不是 None），方便后面链式使用结果。"""
    result = await node.run(state, ctx)
    assert result is not None
    return dict(result)


def test_render_fills_missing_fields_with_blank_and_joins_lists():
    assert render("{a}|{missing}|{items}", {"a": 1, "items": ["x", "y"]}) == "1||x\n\ny"


async def test_function_node_supports_sync_and_async():
    async def async_fn(state):
        return {"y": state["x"] + 1}

    assert await FunctionNode("s", lambda s: {"y": 1}).run({}, CTX) == {"y": 1}
    assert await FunctionNode("a", async_fn).run({"x": 1}, CTX) == {"y": 2}


async def test_agent_node_renders_prompt_and_labels_output():
    llm = ScriptedLLM(ModelTurn(content="建议内容"))
    registry = AgentRegistry()
    await registry.register(Agent("helper", "你是助手", llm))
    node = AgentNode("n", registry, "helper", "请分析：{input}", "opinions", label="专家")
    assert await node.run({"input": "睡眠"}, CTX) == {"opinions": "【专家】建议内容"}
    assert llm.calls[0][-1].content == "请分析：睡眠"


async def test_tool_node_success_and_failure():
    node = ToolNode("calc", CalculatorTool(), lambda s: {"expression": s["input"]}, "calc")
    assert await node.run({"input": "6*7"}, CTX) == {"calc": "6*7 = 42"}
    with pytest.raises(RuntimeError, match="division by zero"):
        await node.run({"input": "1/0"}, CTX)


async def test_router_node_uses_keywords_then_default():
    node = RouterNode("r", {"kids": ["孩子"], "tech": ["Python"]}, default="other")
    assert await node.run({"input": "孩子不睡觉"}, CTX) == {"route": "kids"}
    assert await node.run({"input": "python 报错"}, CTX) == {"route": "tech"}  # 不区分大小写
    assert await node.run({"input": "今天吃什么"}, CTX) == {"route": "other"}


async def test_loop_node_counts_and_stops():
    node = LoopNode("loop", lambda s: True, max_rounds=2)
    assert await node.run({}, CTX) == {"round": 1, "loop": "continue"}
    assert await node.run({"round": 1}, CTX) == {"round": 2, "loop": "done"}
    stop_early = LoopNode("loop", lambda s: False)
    assert (await run(stop_early, {}))["loop"] == "done"


async def test_supervisor_without_llm_dispatches_each_worker_once():
    node = SupervisorNode("boss", ["a", "b"])
    first = await run(node, {})
    assert first == {"next": "a", "dispatched": ["a"]}
    second = await run(node, first)
    assert second == {"next": "b", "dispatched": ["a", "b"]}
    assert (await run(node, second))["next"] == FINISH


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("交给 b 吧", "b"), ("已经足够了，FINISH", FINISH), ("我也不知道", "a")],
)
async def test_supervisor_follows_llm_and_falls_back(reply, expected):
    node = SupervisorNode("boss", ["a", "b"], llm=ScriptedLLM(ModelTurn(content=reply)))
    assert (await run(node, {"input": "任务"}))["next"] == expected


async def test_supervisor_round_limit():
    node = SupervisorNode("boss", ["a"], max_rounds=1)
    assert (await run(node, {"dispatched": ["a"]}))["next"] == FINISH


async def test_human_approval_interrupts_first_then_reads_decision():
    node = HumanApprovalNode("approval", "发布吗？", preview_key="draft")
    with pytest.raises(Interrupt) as raised:
        await node.run({"draft": "草稿"}, CTX)
    assert raised.value.payload == {"question": "发布吗？", "preview": "草稿"}

    rejected = NodeContext("t", 1, resume={"approved": False, "comment": "太长"})
    assert await node.run({}, rejected) == {"approved": False, "comment": "太长"}
    approved = NodeContext("t", 1, resume=True)
    assert await node.run({}, approved) == {"approved": True, "comment": ""}
