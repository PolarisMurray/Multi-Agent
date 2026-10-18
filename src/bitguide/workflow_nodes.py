"""常用节点。每个节点只做一件事：读 State → 返回要写入的字段。

FunctionNode       任意 Python 函数（同步 / 异步都行）
AgentNode          从 AgentRegistry 借一个 Agent 跑一次 ReAct
ToolNode           直接调用一个工具（不经过 LLM）
RouterNode         关键词分类，结果写进 state["route"]，配合条件边使用
LoopNode           循环计数 + 终止条件，防止“写稿-评审”无限循环
SupervisorNode     主管：决定下一个由哪个 worker 干活，或者 FINISH
HumanApprovalNode  第一次执行时抛 Interrupt 暂停；恢复时拿到人工输入继续
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import LLMClient, Tool
from .models import AgentRequest, Message
from .registry import AgentRegistry
from .workflow_core import NodeContext, State
from .workflow_runtime import Interrupt

FINISH = "FINISH"
Updates = Mapping[str, Any] | None


class _Blank(dict[str, Any]):
    """模板里引用了 State 中不存在的字段时，替换成空字符串而不是报 KeyError。"""

    def __missing__(self, key: str) -> str:
        return ""


def render(template: str, state: State) -> str:
    """用 State 填模板，例如 "请评审：{draft}"；列表字段会按段落拼接。"""
    values = {k: "\n\n".join(map(str, v)) if isinstance(v, list) else v for k, v in state.items()}
    return template.format_map(_Blank(values))


@dataclass(slots=True)
class FunctionNode:
    name: str
    fn: Callable[[State], Updates | Awaitable[Updates]]

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        result = self.fn(state)
        return await result if inspect.isawaitable(result) else result


@dataclass(slots=True)
class AgentNode:
    name: str
    agents: AgentRegistry
    agent_id: str
    prompt: str = "{input}"
    output_key: str = "output"
    label: str | None = None  # 设置后输出变成 "【label】内容"，方便汇总时区分是谁说的

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        message = render(self.prompt, state)
        # 每个线程 + 节点一个会话：循环里同一个节点能“记得”自己上一轮写了什么
        request = AgentRequest(message, session_id=f"workflow:{ctx.thread_id}:{self.name}")
        # 用 acquire 借用：执行途中 Agent 被热替换也不会被关掉
        async with self.agents.acquire(self.agent_id) as agent:
            response = await agent.run(request)
        content = f"【{self.label}】{response.content}" if self.label else response.content
        return {self.output_key: content}


@dataclass(slots=True)
class ToolNode:
    name: str
    tool: Tool
    arguments: Callable[[State], Mapping[str, Any]]
    output_key: str = "tool_result"

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        result = await self.tool.invoke(self.arguments(state))
        if not result.success:
            # 工具失败 = 节点失败 → 工作流进入 failed，修好后可以 resume 重试
            raise RuntimeError(f"tool {self.tool.name} failed: {result.error}")
        return {self.output_key: result.content}


@dataclass(slots=True)
class RouterNode:
    name: str
    routes: Mapping[str, list[str]]  # 分支名 → 触发关键词
    default: str
    input_key: str = "input"
    output_key: str = "route"

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        # TODO(改写): 关键词都没命中时，可以再问一次 LLM 让它分类
        text = str(state.get(self.input_key, "")).casefold()
        for route, keywords in self.routes.items():
            if any(word.casefold() in text for word in keywords):
                return {self.output_key: route}
        return {self.output_key: self.default}


@dataclass(slots=True)
class LoopNode:
    name: str
    should_continue: Callable[[State], bool]
    max_rounds: int = 3
    counter_key: str = "round"
    output_key: str = "loop"

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        rounds = int(state.get(self.counter_key, 0)) + 1
        go_on = rounds < self.max_rounds and self.should_continue(state)
        return {self.counter_key: rounds, self.output_key: "continue" if go_on else "done"}


@dataclass(slots=True)
class SupervisorNode:
    """有 LLM 时让 LLM 决定下一步；没有 LLM 或回答无法解析时，按顺序派给还没干过的 worker。"""

    name: str
    workers: list[str]
    llm: LLMClient | None = None
    input_key: str = "input"
    notes_key: str = "notes"
    output_key: str = "next"
    dispatched_key: str = "dispatched"
    max_rounds: int = 6

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        dispatched = list(state.get(self.dispatched_key, []))
        if len(dispatched) >= self.max_rounds:
            return {self.output_key: FINISH}
        choice = await self._ask_llm(state, dispatched) if self.llm else None
        if choice is None:
            remaining = [w for w in self.workers if w not in dispatched]
            choice = remaining[0] if remaining else FINISH
        if choice != FINISH:
            dispatched.append(choice)
        return {self.output_key: choice, self.dispatched_key: dispatched}

    async def _ask_llm(self, state: State, dispatched: list[str]) -> str | None:
        assert self.llm is not None
        prompt = (
            f"你是任务主管。用户任务：{state.get(self.input_key, '')}\n"
            f"可选成员：{', '.join(self.workers)}\n已派过：{', '.join(dispatched) or '无'}\n"
            f"已有成果：\n{render('{' + self.notes_key + '}', state) or '无'}\n"
            f"下一步交给谁？只回答一个成员名；如果已经足够，回答 {FINISH}。"
        )
        turn = await self.llm.complete([Message("user", prompt)], [])
        answer = turn.content.strip()
        if FINISH in answer.upper():
            return FINISH
        # 在回答里找第一个出现的成员名；找不到就返回 None 交给兜底规则
        found = [(answer.find(w), w) for w in self.workers if w in answer]
        return min(found)[1] if found else None


@dataclass(slots=True)
class HumanApprovalNode:
    name: str
    question: str = "请审批以下内容"
    preview_key: str = "draft"
    output_key: str = "approved"
    comment_key: str = "comment"

    async def run(self, state: State, ctx: NodeContext) -> Updates:
        if ctx.resume is None:
            # 第一次执行到这里：暂停整个工作流，把要审的内容交给人
            raise Interrupt({"question": self.question, "preview": state.get(self.preview_key)})
        decision = ctx.resume if isinstance(ctx.resume, Mapping) else {"approved": bool(ctx.resume)}
        return {
            self.output_key: bool(decision.get("approved")),
            self.comment_key: str(decision.get("comment", "")),
        }
