"""可观测性：Agent 和工作流在关键节点 emit 事件，Tracer 决定怎么展示。

- NullTracer：什么都不做（默认）。
- ConsoleTracer：“学习模式”，把 推理 → 调工具 → 观察 → 回答 打印到终端。
- JsonLogTracer：每个事件一行 JSON，方便接 ELK / Loki 之类的日志系统。
- RecordingTracer：把事件存进列表，测试里用来断言。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TextIO


@dataclass(slots=True, frozen=True)
class TraceEvent:
    kind: str  # 例如 "llm.turn"、"tool.result"、"workflow.step"
    source: str  # 发出事件的 Agent id 或工作流名
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class Tracer(Protocol):
    def emit(self, kind: str, source: str, **data: Any) -> None: ...


class NullTracer:
    def emit(self, kind: str, source: str, **data: Any) -> None:
        return None


class RecordingTracer:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, kind: str, source: str, **data: Any) -> None:
        self.events.append(TraceEvent(kind, source, data))

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


class JsonLogTracer:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr

    def emit(self, kind: str, source: str, **data: Any) -> None:
        event = TraceEvent(kind, source, data)
        record = {"ts": event.timestamp, "kind": kind, "source": source, **data}
        print(json.dumps(record, ensure_ascii=False, default=str), file=self._stream, flush=True)


class ConsoleTracer:
    """学习模式：每个事件一行中文说明。默认写到 stderr，不污染 stdout 上的最终答案。"""

    def __init__(self, stream: TextIO | None = None, max_length: int = 300) -> None:
        self._stream = stream or sys.stderr
        self._max = max_length

    def emit(self, kind: str, source: str, **data: Any) -> None:
        line = self._format(kind, data)
        if line is not None:
            print(f"  [{source}] {line}", file=self._stream, flush=True)

    def _cut(self, value: Any) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= self._max else text[: self._max] + "…"

    def _format(self, kind: str, d: dict[str, Any]) -> str | None:
        # TODO(改写): 换成 rich 之类的库可以做出带颜色和面板的输出
        match kind:
            case "agent.start":
                return f"▶ 用户：{self._cut(d['message'])}"
            case "skills.matched":
                return f"🧩 启用技能：{'、'.join(d['skills'])}"
            case "rag.retrieved":
                return f"📚 检索到 {d['count']} 段资料"
            case "llm.turn" if d["tool_calls"]:
                calls = ", ".join(
                    f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})"
                    for c in d["tool_calls"]
                )
                return f"🤔 第 {d['iteration']} 轮推理 → 调用工具 {self._cut(calls)}"
            case "tool.result":
                mark = "✓" if d["success"] else "✗"
                return f"👀 观察 {d['name']} {mark} {self._cut(d['content'])}"
            case "agent.answer":
                return f"✅ 回答（共 {d['iterations']} 轮）"
            case "agent.degraded":
                return f"⚠️  {d['component']} 不可用，已降级：{self._cut(d['error'])}"
            case "workflow.start":
                return f"🚀 开始运行，thread={d['thread_id']}"
            case "workflow.step":
                return f"⏩ 第 {d['step']} 步并行执行：{', '.join(d['nodes'])}"
            case "workflow.node_done":
                return f"   · {d['node']} 完成，写入 {', '.join(d['keys']) or '无'}"
            case "workflow.interrupted":
                return f"⏸  在 {d['node']} 暂停，等待人工输入"
            case "workflow.finish":
                return f"🏁 结束：{d['status']}（{d['steps']} 步）"
            case _:
                return None
