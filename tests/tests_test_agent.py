import pytest

from bitguide.agent import Agent
from bitguide.agent import ConversationMemory
from bitguide.agent import AgentRequest, ModelTurn, ToolCall
from bitguide.agent import CalculatorTool

class ScriptedLLM:
    """return each ModelTurn by steps, and record, save the tools and messages that each call used"""

    def __init__ (self, *turn: ModelTurn):
        self._turns = list(turns)
        self.call : list[list] = []
        self.tools_seen : list[list] = []

    async def complete(self, messages, tools):
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        return self._turns.pop(0)



class FixedRetriever:
    async def retrieve(self, query, top_k = 5):
        return ["5 year old kid should not plauy cod"]


def calc(call_id :str, expr : str) -> ToolCall:
    return ToolCall(call_id, "calculator", {"expression": expr})

def req(text: str, session: str = "s1") -> AgentRequest:
    return AgentRequest(message = text, sesion_id = session)

async def test_direct_answer_without_tools():
    llm = ScriptedLLM(ModelTurn(content="你好！"))
    memory = ConversationMemory()
    agent = Agent("a1", "你是助手", llm, memory=memory)

    resp = await agent.run(req("hi"))

    assert (resp.content, resp.agent_id, resp.session_id, resp.iterations) == ("你好！", "a1", "s1", 1)
    assert [(m.role, m.content) for m in llm.calls[0]] == [("system", "你是助手"), ("user", "hi")]
    assert [(i.type, i.content) for i in await memory.history("s1")] == [
        ("user", "hi"),
        ("assistant", "你好！"),
    ]


async def test_tool_call_then_answer():
    llm = ScriptedLLM(
        ModelTurn(tool_calls=[calc("c1", "3 * 7")]),
        ModelTurn(content="答案是 21"),
    )
    agent = Agent("a1", "sys", llm, tools=[CalculatorTool()])

    resp = await agent.run(req("3 乘 7？"))

    assert resp.content == "答案是 21" and resp.iterations == 2
    second = llm.calls[1]
    assert [m.role for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2].tool_calls[0].id == "c1"
    assert second[3].tool_call_id == "c1"
    assert second[3].name == "calculator"
    assert second[3].content.endswith("21")


async def test_multiple_tool_calls_keep_order():
    llm = ScriptedLLM(
        ModelTurn(tool_calls=[calc("c1", "1 + 1"), calc("c2", "2 + 2")]),
        ModelTurn(content="done"),
    )
    agent = Agent("a1", "sys", llm, tools=[CalculatorTool()])
    await agent.run(req("q"))
    tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]


async def test_tool_failure_becomes_observation():
    llm = ScriptedLLM(
        ModelTurn(tool_calls=[ToolCall("c1", "no_such_tool", {})]),
        ModelTurn(content="抱歉，工具不可用"),
    )
    agent = Agent("a1", "sys", llm)
    resp = await agent.run(req("q"))
    assert resp.content == "抱歉，工具不可用"
    assert "does not exist" in llm.calls[1][-1].content


async def test_tool_schemas_sent_to_model():
    llm = ScriptedLLM(ModelTurn(content="ok"))
    agent = Agent("a1", "sys", llm, tools=[CalculatorTool()])
    await agent.run(req("q"))
    (schema,) = llm.tools_seen[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert schema["function"]["parameters"]["required"] == ["expression"]


async def test_history_is_sent_but_tool_messages_are_filtered():
    memory = ConversationMemory()
    llm = ScriptedLLM(
        ModelTurn(tool_calls=[calc("c1", "3 * 7")]),
        ModelTurn(content="21"),
        ModelTurn(content="不客气"),
    )
    agent = Agent("a1", "sys", llm, tools=[CalculatorTool()], memory=memory)

    await agent.run(req("3 乘 7？"))
    await agent.run(req("谢谢"))

    assert [(m.role, m.content) for m in llm.calls[2]] == [
        ("system", "sys"),
        ("user", "3 乘 7？"),
        ("assistant", "21"),
        ("user", "谢谢"),
    ]
    # 工具记录仍然保存在记忆里，用于审计
    assert any(i.type == "tool" for i in await memory.history("s1"))


async def test_retriever_context_is_added_to_system_prompt():
    llm = ScriptedLLM(ModelTurn(content="ok"))
    agent = Agent("a1", "sys", llm, retriever=FixedRetriever())
    await agent.run(req("玩具推荐"))
    system = llm.calls[0][0]
    assert system.role == "system"
    assert system.content.startswith("sys")
    assert "拼图和积木" in system.content


async def test_exceeding_max_iterations_raises_and_saves_nothing():
    memory = ConversationMemory()
    llm = ScriptedLLM(*[ModelTurn(tool_calls=[calc(f"c{i}", "1 + 1")]) for i in range(3)])
    agent = Agent("a1", "sys", llm, tools=[CalculatorTool()], max_iterations=3, memory=memory)

    with pytest.raises(RuntimeError):
        await agent.run(req("loop"))

    assert len(llm.calls) == 3
    assert all(i.type == "tool" for i in await memory.history("s1"))  # 没写入 user / assistant


async def test_empty_answer_raises():
    agent = Agent("a1", "sys", ScriptedLLM(ModelTurn(content="   ")))
    with pytest.raises(RuntimeError):
        await agent.run(req("q"))


def test_rejects_non_positive_max_iterations():
    with pytest.raises(ValueError):
        Agent("a1", "sys", ScriptedLLM(), max_iterations=0)