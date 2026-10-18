"""工作流注册表：名字 → 已校验的 Graph。"""

from __future__ import annotations

from .workflow_core import Graph


class WorkflowRegistry:
    def __init__(self) -> None:
        self._graphs: dict[str, Graph] = {}

    def register(self, graph: Graph, *, replace: bool = False) -> None:
        graph.validate()  # 注册时就校验，坏掉的图进不了系统
        if graph.name in self._graphs and not replace:
            raise ValueError(f"workflow already registered: {graph.name}")
        self._graphs[graph.name] = graph

    def get(self, name: str) -> Graph:
        try:
            return self._graphs[name]
        except KeyError:
            raise KeyError(f"workflow not found: {name}") from None

    def names(self) -> list[str]:
        return sorted(self._graphs)

    def all(self) -> list[Graph]:
        return [self._graphs[name] for name in self.names()]
