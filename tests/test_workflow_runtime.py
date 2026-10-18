import asyncio

import pytest

from bitguide.observability import RecordingTracer
from bitguide.workflow_checkpoint import FileCheckpointStore, MemoryCheckpointStore
from bitguide.workflow_core import END, START, Append, Graph
from bitguide.workflow_nodes import FunctionNode, HumanApprovalNode
from bitguide.workflow_runtime import WorkflowRunner


def runner(graph: Graph, **kwargs) -> WorkflowRunner:
    return WorkflowRunner(graph, kwargs.pop("store", MemoryCheckpointStore()), **kwargs)


async def test_sequential_run_saves_checkpoint():
    store = MemoryCheckpointStore()
    graph = (
        Graph("seq")
        .add_node(FunctionNode("double", lambda s: {"x": s["x"] * 2}))
        .add_node(FunctionNode("plus", lambda s: {"x": s["x"] + 1, "output": s["x"] + 1}))
        .add_edge(START, "double")
        .add_edge("double", "plus")
        .add_edge("plus", END)
    )
    result = await runner(graph, store=store).run({"x": 5}, "t1")
    assert (result.status, result.output, result.steps) == ("completed", 11, 2)
    saved = await store.load("t1")
    assert saved is not None and saved.status == "completed" and saved.state["x"] == 11


def fan_out_graph(names: list[str], probe) -> Graph:
    graph = Graph("fan").add_channel("seen", Append())
    for name in names:
        graph.add_node(FunctionNode(name, probe(name))).add_edge(START, name).add_edge(name, "join")
    return graph.add_node(FunctionNode("join", lambda s: {"output": s["seen"]})).add_edge(
        "join", END
    )


async def test_parallel_nodes_share_one_snapshot_and_merge_in_frontier_order():
    def probe(name):
        async def run(state):
            await asyncio.sleep(0.03 if name == "a" else 0)  # a 后完成，但合并顺序仍然是 a 在前
            return {"seen": f"{name}:{state.get('counter', 0)}", "counter_" + name: 1}

        return run

    result = await runner(fan_out_graph(["a", "b"], probe)).run({"counter": 0})
    assert result.output == ["a:0", "b:0"]  # 两个节点都看到了同一份旧值


async def test_parallelism_is_bounded_by_the_semaphore():
    running = peak = 0

    def probe(name):
        async def run(state):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return {"seen": name}

        return run

    result = await runner(fan_out_graph(["a", "b", "c", "d"], probe), parallelism=2).run({})
    assert result.status == "completed"
    assert peak == 2


async def test_conflicting_overwrites_fail_the_run():
    graph = Graph("conflict")
    for name in ("a", "b"):
        graph.add_node(FunctionNode(name, lambda s: {"x": 1})).add_edge(START, name).add_edge(
            name, END
        )
    result = await runner(graph).run({})
    assert result.status == "failed" and "Append" in (result.error or "")


def counter_loop() -> Graph:
    return (
        Graph("loop")
        .add_node(FunctionNode("inc", lambda s: {"n": s.get("n", 0) + 1}))
        .add_edge(START, "inc")
        .add_conditional_edges(
            "inc", lambda s: "again" if s["n"] < 3 else "stop", {"again": "inc", "stop": END}
        )
    )


async def test_cycles_run_until_condition_and_max_steps_stops_runaway_loops():
    assert (await runner(counter_loop()).run({})).state["n"] == 3
    stopped = await runner(counter_loop(), max_steps=2).run({})
    assert stopped.status == "failed" and "max_steps" in (stopped.error or "")


def approval_graph(calls: dict[str, int]) -> Graph:
    def draft(state):
        calls["draft"] = calls.get("draft", 0) + 1
        return {"draft": "草稿"}

    return (
        Graph("approve")
        .add_node(FunctionNode("draft", draft))
        .add_node(HumanApprovalNode("approval"))
        .add_node(FunctionNode("publish", lambda s: {"output": f"发布：{s['draft']}"}))
        .add_edge(START, "draft")
        .add_edge("draft", "approval")
        .add_edge("approval", "publish")
        .add_edge("publish", END)
    )


async def test_interrupt_then_resume_in_a_new_runner(tmp_path):
    calls: dict[str, int] = {}
    first = await WorkflowRunner(approval_graph(calls), FileCheckpointStore(tmp_path)).run({}, "t")
    assert first.status == "interrupted"
    assert first.interrupt == {
        "node": "approval",
        "payload": {"question": "请审批以下内容", "preview": "草稿"},
    }

    # 新的 runner + 新的 store 对象，模拟“进程重启后恢复”
    restarted = WorkflowRunner(approval_graph(calls), FileCheckpointStore(tmp_path))
    final = await restarted.resume("t", {"approved": True, "comment": "好"})
    assert final.status == "completed"
    assert final.output == "发布：草稿"
    assert final.state["comment"] == "好"
    assert calls["draft"] == 1  # 已经完成的步骤不会重跑


async def test_sibling_writes_survive_an_interrupt_in_the_same_step():
    calls = {"worker": 0}

    def worker(state):
        calls["worker"] += 1
        return {"work": "done"}

    graph = (
        Graph("mixed")
        .add_node(HumanApprovalNode("approval"))
        .add_node(FunctionNode("worker", worker))
        .add_edge(START, "approval")
        .add_edge(START, "worker")
        .add_edge("approval", END)
        .add_edge("worker", END)
    )
    store = MemoryCheckpointStore()
    paused = await runner(graph, store=store).run({}, "t")
    assert paused.status == "interrupted"
    saved = await store.load("t")
    assert saved is not None and saved.pending_writes == {"worker": {"work": "done"}}
    final = await runner(graph, store=store).resume("t", True)
    assert final.state["work"] == "done" and final.state["approved"] is True
    assert calls["worker"] == 1


async def test_failed_node_is_retried_without_rerunning_successful_siblings():
    calls = {"flaky": 0, "steady": 0}

    def flaky(state):
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise ConnectionError("model timeout")
        return {"a": 1}

    def steady(state):
        calls["steady"] += 1
        return {"b": 2}

    graph = (
        Graph("retry")
        .add_node(FunctionNode("flaky", flaky))
        .add_node(FunctionNode("steady", steady))
        .add_edge(START, "flaky")
        .add_edge(START, "steady")
        .add_edge("flaky", END)
        .add_edge("steady", END)
    )
    store = MemoryCheckpointStore()
    failed = await runner(graph, store=store).run({}, "t")
    assert failed.status == "failed" and "ConnectionError" in (failed.error or "")
    retried = await runner(graph, store=store).resume("t")
    assert retried.status == "completed" and retried.state == {"a": 1, "b": 2}
    assert calls == {"flaky": 2, "steady": 1}


async def test_resume_guards():
    store = MemoryCheckpointStore()
    done = await runner(counter_loop(), store=store).run({}, "done")
    with pytest.raises(KeyError):
        await runner(counter_loop(), store=store).resume("nope")
    with pytest.raises(ValueError, match="completed"):
        await runner(counter_loop(), store=store).resume(done.thread_id)
    with pytest.raises(ValueError, match="belongs to"):
        await runner(approval_graph({}), store=store).resume(done.thread_id)


async def test_node_must_return_a_mapping():
    bad = FunctionNode("n", lambda s: 42)  # type: ignore[arg-type]  —— 故意返回错误类型
    graph = Graph("bad").add_node(bad).add_edge(START, "n").add_edge("n", END)
    result = await runner(graph).run({})
    assert result.status == "failed" and "mapping" in (result.error or "")


async def test_tracer_sees_steps_and_finish():
    tracer = RecordingTracer()
    await runner(counter_loop(), tracer=tracer).run({})
    assert tracer.kinds()[0] == "workflow.start"
    assert tracer.kinds().count("workflow.step") == 3
    assert tracer.kinds()[-1] == "workflow.finish"
