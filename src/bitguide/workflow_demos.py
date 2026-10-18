"""8 个演示工作流：同一套引擎，覆盖顺序、分支、并行、循环、主管调度、人工审批。

约定：输入放在 state["input"]，最终结果放在 state["output"]。
用到的 Agent（见 config/platform.yaml）：parenting_advisor、tech_helper、content_editor。
"""

from __future__ import annotations

from .contracts import LLMClient
from .registry import AgentRegistry
from .tools import CalculatorTool
from .workflow_core import END, START, Append, Graph, State
from .workflow_nodes import (
    FINISH,
    AgentNode,
    FunctionNode,
    HumanApprovalNode,
    LoopNode,
    RouterNode,
    SupervisorNode,
    ToolNode,
)

PARENTING, TECH, EDITOR = "parenting_advisor", "tech_helper", "content_editor"


def pipeline(agents: AgentRegistry) -> Graph:
    """① 顺序管线：预处理 → Agent → 后处理。"""

    def prepare(state: State):
        return {"question": f"{state['input'].strip()}（请分点回答，不超过 5 点）"}

    def finalize(state: State):
        return {"output": f"📝 {state['answer']}\n—— 由 pipeline 工作流生成"}

    return (
        Graph("pipeline", "顺序管线：预处理 → 亲子顾问 → 格式化")
        .add_node(FunctionNode("prepare", prepare))
        .add_node(AgentNode("advisor", agents, PARENTING, "{question}", "answer"))
        .add_node(FunctionNode("finalize", finalize))
        .add_edge(START, "prepare")
        .add_edge("prepare", "advisor")
        .add_edge("advisor", "finalize")
        .add_edge("finalize", END)
    )


def router(agents: AgentRegistry) -> Graph:
    """② 条件路由：按关键词把问题分给最合适的 Agent。"""
    routes = {
        "parenting": ["孩子", "宝宝", "育儿", "亲子"],
        "tech": ["代码", "程序", "bug", "python", "接口", "数据库"],
    }
    return (
        Graph("router", "条件路由：关键词分类 → 亲子 / 技术 / 通用")
        .add_node(RouterNode("classify", routes, default="general"))
        .add_node(AgentNode("parenting", agents, PARENTING))
        .add_node(AgentNode("tech", agents, TECH))
        .add_node(AgentNode("general", agents, EDITOR))
        .add_edge(START, "classify")
        .add_conditional_edges(
            "classify",
            lambda s: s["route"],
            {"parenting": "parenting", "tech": "tech", "general": "general"},
        )
        .add_edge("parenting", END)
        .add_edge("tech", END)
        .add_edge("general", END)
    )


def _expert_nodes(graph: Graph, agents: AgentRegistry) -> Graph:
    """两位专家并行作答，写进同一个 Append 字段 opinions。"""
    prompt = "请从你的专业角度，用 3 句话分析：{input}"
    return (
        graph.add_channel("opinions", Append())
        .add_node(AgentNode("parenting", agents, PARENTING, prompt, "opinions", label="亲子顾问"))
        .add_node(AgentNode("tech", agents, TECH, prompt, "opinions", label="技术助手"))
        .add_edge(START, "parenting")  # START 有两条出边 = 并行分叉
        .add_edge(START, "tech")
    )


def parallel_experts(agents: AgentRegistry) -> Graph:
    """③ 并行多专家：fan-out 两位专家 → fan-in 编辑汇总。"""
    graph = _expert_nodes(Graph("parallel_experts", "并行多专家 → 编辑汇总"), agents)
    return (
        graph.add_node(
            AgentNode("editor", agents, EDITOR, "请把以下专家意见合并成一份建议：\n{opinions}")
        )
        .add_edge("parenting", "editor")  # 两条边指向同一个节点 = 汇合，editor 只执行一次
        .add_edge("tech", "editor")
        .add_edge("editor", END)
    )


def review_loop(agents: AgentRegistry) -> Graph:
    """④ 循环：写稿 → 评审 → 不通过就带着意见重写，最多 3 轮。"""
    write_prompt = "请就「{input}」写一段 100 字以内的建议。\n上一轮评审意见（如有）：{review}"
    review_prompt = "请评审下面的稿子。合格只回复“通过”，否则给出一条修改意见：\n{draft}"
    return (
        Graph("review_loop", "写稿-评审循环（LoopNode 防止无限循环）")
        .add_node(AgentNode("writer", agents, PARENTING, write_prompt, "draft"))
        .add_node(AgentNode("reviewer", agents, EDITOR, review_prompt, "review"))
        .add_node(
            LoopNode(
                "loop", lambda s: not s.get("review", "").strip().startswith("通过"), max_rounds=3
            )
        )
        .add_node(FunctionNode("finalize", lambda s: {"output": s["draft"]}))
        .add_edge(START, "writer")
        .add_edge("writer", "reviewer")
        .add_edge("reviewer", "loop")
        .add_conditional_edges(
            "loop", lambda s: s["loop"], {"continue": "writer", "done": "finalize"}
        )
        .add_edge("finalize", END)
    )


def supervisor(agents: AgentRegistry, llm: LLMClient | None) -> Graph:
    """⑤ 主管调度：Supervisor 反复决定“下一个谁来做”，直到 FINISH。"""
    prompt = "主管派给你的任务：{input}\n已有同事的成果：\n{notes}"
    workers = ["parenting", "tech"]
    return (
        Graph("supervisor", "主管 Agent 动态调度多个 worker")
        .add_channel("notes", Append())
        .add_node(SupervisorNode("supervisor", workers, llm))
        .add_node(AgentNode("parenting", agents, PARENTING, prompt, "notes", label="亲子顾问"))
        .add_node(AgentNode("tech", agents, TECH, prompt, "notes", label="技术助手"))
        .add_node(FunctionNode("summarize", lambda s: {"output": "\n\n".join(s.get("notes", []))}))
        .add_edge(START, "supervisor")
        .add_conditional_edges(
            "supervisor",
            lambda s: s["next"],
            {"parenting": "parenting", "tech": "tech", FINISH: "summarize"},
        )
        .add_edge("parenting", "supervisor")  # worker 做完回到主管 → 形成循环
        .add_edge("tech", "supervisor")
        .add_edge("summarize", END)
    )


def human_approval(agents: AgentRegistry) -> Graph:
    """⑥ 人工审批（HITL）：起草 → 暂停等审批 → 通过就发布，驳回就按意见修改后再审。"""
    draft_prompt = (
        "请为「{input}」写一份简短的通知草稿。\n审批意见（如有）：{comment}\n上一版：{draft}"
    )
    return (
        Graph("human_approval", "起草 → 人工审批（中断/恢复）→ 发布或修改")
        .add_node(AgentNode("drafter", agents, EDITOR, draft_prompt, "draft"))
        .add_node(HumanApprovalNode("approval", "是否发布这份通知？"))
        .add_node(FunctionNode("publish", lambda s: {"output": f"✅ 已发布：\n{s['draft']}"}))
        .add_edge(START, "drafter")
        .add_edge("drafter", "approval")
        .add_conditional_edges(
            "approval",
            lambda s: "yes" if s["approved"] else "no",
            {"yes": "publish", "no": "drafter"},
        )
        .add_edge("publish", END)
    )


def tool_pipeline(agents: AgentRegistry) -> Graph:
    """⑦ 工具节点：先确定性地算出结果，再让 Agent 解释（不让 LLM 自己算数）。"""
    return (
        Graph("tool_pipeline", "计算器工具节点 → Agent 解释结果")
        .add_node(
            ToolNode("calculate", CalculatorTool(), lambda s: {"expression": s["input"]}, "calc")
        )
        .add_node(AgentNode("explain", agents, TECH, "请用一句话解释这个计算：{calc}"))
        .add_edge(START, "calculate")
        .add_edge("calculate", "explain")
        .add_edge("explain", END)
    )


def expert_review(agents: AgentRegistry) -> Graph:
    """⑧ 综合：并行专家 → 编辑汇总 → 人工审批 → 发布（驳回则编辑按意见重写）。"""
    edit_prompt = "请把专家意见合并成一份方案：\n{opinions}\n审批意见（如有）：{comment}"
    graph = _expert_nodes(Graph("expert_review", "并行专家 → 编辑汇总 → 人工审批 → 发布"), agents)
    return (
        graph.add_node(AgentNode("editor", agents, EDITOR, edit_prompt, "draft"))
        .add_node(HumanApprovalNode("approval", "方案是否可以发布？"))
        .add_node(FunctionNode("publish", lambda s: {"output": s["draft"]}))
        .add_edge("parenting", "editor")
        .add_edge("tech", "editor")
        .add_edge("editor", "approval")
        .add_conditional_edges(
            "approval",
            lambda s: "yes" if s["approved"] else "no",
            {"yes": "publish", "no": "editor"},
        )
        .add_edge("publish", END)
    )


def build_demo_workflows(
    agents: AgentRegistry, supervisor_llm: LLMClient | None = None
) -> list[Graph]:
    return [
        pipeline(agents),
        router(agents),
        parallel_experts(agents),
        review_loop(agents),
        supervisor(agents, supervisor_llm),
        human_approval(agents),
        tool_pipeline(agents),
        expert_review(agents),
    ]
