"""并行加速基准：4 个“专家”节点各模拟 200ms 的模型调用，对比串行与并行的总耗时。

运行：uv run python scripts/bench_parallel.py
"""

import asyncio
import time

from bitguide.workflow_checkpoint import MemoryCheckpointStore
from bitguide.workflow_core import END, START, Append, Graph
from bitguide.workflow_nodes import FunctionNode
from bitguide.workflow_runtime import WorkflowRunner

EXPERTS = ["e1", "e2", "e3", "e4"]
LATENCY = 0.2


def expert(name: str):
    async def run(state):
        await asyncio.sleep(LATENCY)  # 模拟一次模型调用的网络等待
        return {"opinions": name}

    return run


def build_graph() -> Graph:
    graph = Graph("bench").add_channel("opinions", Append())
    for name in EXPERTS:
        graph.add_node(FunctionNode(name, expert(name))).add_edge(START, name).add_edge(
            name, "join"
        )
    join = FunctionNode("join", lambda s: {"output": s["opinions"]})
    return graph.add_node(join).add_edge("join", END)


async def main() -> None:
    timings = {}
    for parallelism in (1, len(EXPERTS)):
        start = time.perf_counter()
        await WorkflowRunner(build_graph(), MemoryCheckpointStore(), parallelism=parallelism).run(
            {}
        )
        timings[parallelism] = time.perf_counter() - start
        print(f"parallelism={parallelism}: {timings[parallelism]:.3f}s")
    saved = 1 - timings[len(EXPERTS)] / timings[1]
    print(f"并行节省耗时 {saved:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
