"""工作流引擎的数据结构：State / Channel / Node / Edge / Graph。

心智模型（和 LangGraph / Pregel 一致）：
- State 是一块共享“黑板”（dict）。节点只读 State，返回一个“想写入的字段”字典。
- Channel 决定同一个字段被写入时怎么合并：覆盖、追加、还是只活一步。
- Edge 决定一个节点跑完后下一步跑谁；一个节点有多条出边 = 并行分叉（fan-out），
  多个节点指向同一个节点 = 汇合（fan-in），下一步只会执行一次。
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

START = "__start__"
END = "__end__"

State = Mapping[str, Any]
Router = Callable[[State], str | list[str]]


class InvalidUpdateError(ValueError):
    """同一步里出现了无法合并的写入，例如两个并行节点同时覆盖同一个字段。"""


# ---------------------------------------------------------------- Channel


class Channel(Protocol):
    @property
    def ephemeral(self) -> bool: ...

    def reduce(self, current: Any, writes: list[Any]) -> Any: ...


@dataclass(slots=True, frozen=True)
class LastValue:
    """覆盖语义（默认）。同一步只允许一个节点写，否则结果取决于谁先跑完，不可预测。"""

    ephemeral: bool = False

    def reduce(self, current: Any, writes: list[Any]) -> Any:
        if len(writes) > 1:
            raise InvalidUpdateError("multiple writes in one step")
        return writes[-1]


@dataclass(slots=True, frozen=True)
class Append:
    """追加语义：并行节点的写入按节点顺序拼成一个列表，适合“收集多位专家的意见”。"""

    ephemeral: bool = False

    def reduce(self, current: Any, writes: list[Any]) -> Any:
        values = list(current or [])
        for value in writes:
            values.extend(value if isinstance(value, list) else [value])
        return values


@dataclass(slots=True, frozen=True)
class Ephemeral:
    """临时语义：写入的值只对紧接着的下一步可见，之后自动清除（例如一次性的指令）。"""

    ephemeral: bool = True

    def reduce(self, current: Any, writes: list[Any]) -> Any:
        if len(writes) > 1:
            raise InvalidUpdateError("multiple writes in one step")
        return writes[-1]


_DEFAULT_CHANNEL = LastValue()


def apply_writes(
    channels: Mapping[str, Channel], state: State, writes: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """把一步里所有节点的写入合并成新的 State（不修改传入的旧 State）。"""
    # 上一步写入的临时字段，到这里就过期了
    new_state = {
        key: value
        for key, value in state.items()
        if not channels.get(key, _DEFAULT_CHANNEL).ephemeral
    }
    grouped: dict[str, list[Any]] = {}
    for update in writes:
        for key, value in update.items():
            grouped.setdefault(key, []).append(copy.deepcopy(value))
    for key, values in grouped.items():
        channel = channels.get(key, _DEFAULT_CHANNEL)
        try:
            new_state[key] = channel.reduce(new_state.get(key), values)
        except InvalidUpdateError as error:
            raise InvalidUpdateError(
                f"state[{key!r}]: {error}. 多个并行节点写同一字段时，请把它声明为 Append Channel"
            ) from error
    return new_state


# ---------------------------------------------------------------- Node


@dataclass(slots=True, frozen=True)
class NodeContext:
    thread_id: str
    step: int
    # 人工输入：只有“被中断、现在恢复执行”的那个节点会拿到非 None 的值
    resume: Any = None


class Node(Protocol):
    @property
    def name(self) -> str: ...

    async def run(self, state: State, ctx: NodeContext) -> Mapping[str, Any] | None: ...


# ---------------------------------------------------------------- Edge


@dataclass(slots=True, frozen=True)
class Edge:
    source: str
    target: str

    def resolve(self, state: State) -> list[str]:
        return [self.target]

    def possible_targets(self) -> set[str]:
        return {self.target}


@dataclass(slots=True, frozen=True)
class ConditionalEdge:
    """router 读 State 返回一个（或多个）分支名，mapping 把分支名翻译成节点名。

    要求必须给 mapping：这样所有可能的去向在运行前就已知，可以做静态校验。
    """

    source: str
    router: Router
    mapping: Mapping[str, str]

    def resolve(self, state: State) -> list[str]:
        chosen = self.router(state)
        branches = [chosen] if isinstance(chosen, str) else list(chosen)
        targets = []
        for branch in branches:
            if branch not in self.mapping:
                raise KeyError(f"router of {self.source!r} returned unknown branch {branch!r}")
            targets.append(self.mapping[branch])
        return targets

    def possible_targets(self) -> set[str]:
        return set(self.mapping.values())


# ---------------------------------------------------------------- Graph


@dataclass(slots=True)
class Graph:
    name: str
    description: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge | ConditionalEdge] = field(default_factory=list)
    channels: dict[str, Channel] = field(default_factory=dict)

    # 所有 add_* 都返回 self，可以链式调用
    def add_node(self, node: Node) -> Graph:
        if node.name in (START, END):
            raise ValueError(f"reserved node name: {node.name}")
        if node.name in self.nodes:
            raise ValueError(f"duplicate node: {node.name}")
        self.nodes[node.name] = node
        return self

    def add_edge(self, source: str, target: str) -> Graph:
        self.edges.append(Edge(source, target))
        return self

    def add_conditional_edges(
        self, source: str, router: Router, mapping: Mapping[str, str]
    ) -> Graph:
        self.edges.append(ConditionalEdge(source, router, dict(mapping)))
        return self

    def add_channel(self, key: str, channel: Channel) -> Graph:
        self.channels[key] = channel
        return self

    def outgoing(self, source: str) -> list[Edge | ConditionalEdge]:
        return [edge for edge in self.edges if edge.source == source]

    def next_nodes(self, sources: Iterable[str], state: State) -> list[str]:
        """计算下一步要执行的节点：去掉 END，去重（fan-in 汇合点只执行一次），保持顺序。"""
        targets = [
            t for source in sources for edge in self.outgoing(source) for t in edge.resolve(state)
        ]
        return [t for t in dict.fromkeys(targets) if t != END]

    def validate(self) -> Graph:
        """运行前的静态检查。图写错了应该在启动时报错，而不是跑到一半才发现。"""
        if not self.nodes:
            raise ValueError(f"{self.name}: graph has no nodes")
        if not self.outgoing(START):
            raise ValueError(f"{self.name}: missing edge from START")
        valid_sources = {*self.nodes, START}
        valid_targets = {*self.nodes, END}
        for edge in self.edges:
            if edge.source not in valid_sources:
                raise ValueError(f"{self.name}: unknown edge source {edge.source!r}")
            for target in edge.possible_targets():
                if target not in valid_targets:
                    raise ValueError(f"{self.name}: unknown edge target {target!r}")
        for name in self.nodes:
            if not self.outgoing(name):
                raise ValueError(f"{self.name}: node {name!r} has no outgoing edge (use END)")

        # BFS：从 START 出发能走到哪些节点
        reachable: set[str] = set()
        queue = deque([START])
        while queue:
            current = queue.popleft()
            for edge in self.outgoing(current):
                for target in edge.possible_targets() - reachable:
                    reachable.add(target)
                    if target != END:
                        queue.append(target)
        unreachable = set(self.nodes) - reachable
        if unreachable:
            raise ValueError(f"{self.name}: unreachable nodes {sorted(unreachable)}")
        if END not in reachable:
            raise ValueError(f"{self.name}: END is unreachable, the workflow could never finish")
        return self
