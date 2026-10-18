import io
import json

from conftest import ScriptedLLM

from bitguide.agent import Agent
from bitguide.models import AgentRequest, ModelTurn, ToolCall
from bitguide.observability import ConsoleTracer, JsonLogTracer, RecordingTracer
from bitguide.tools import CalculatorTool


def test_json_tracer_writes_one_json_object_per_line():
    stream = io.StringIO()
    tracer = JsonLogTracer(stream)
    tracer.emit("tool.result", "agent", name="calculator", success=True)
    tracer.emit("agent.answer", "agent", iterations=2)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r["kind"] for r in records] == ["tool.result", "agent.answer"]
    assert records[0]["name"] == "calculator"
    assert "ts" in records[0]


def test_console_tracer_formats_known_events_and_truncates():
    stream = io.StringIO()
    tracer = ConsoleTracer(stream, max_length=10)
    tracer.emit("tool.result", "a", name="calculator", success=False, content="x" * 50)
    tracer.emit("something.unknown", "a")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert "✗" in lines[0]
    assert "…" in lines[0]


async def test_agent_emits_the_react_trace():
    tracer = RecordingTracer()
    llm = ScriptedLLM(
        ModelTurn(tool_calls=[ToolCall("c1", "calculator", {"expression": "6*7"})]),
        ModelTurn(content="答案是 42"),
    )
    agent = Agent("a", "你是助手", llm, [CalculatorTool()], tracer=tracer)
    await agent.run(AgentRequest("6*7=?", "s"))
    assert tracer.kinds() == ["agent.start", "llm.turn", "tool.result", "llm.turn", "agent.answer"]
    tool_event = tracer.events[2]
    assert tool_event.data["success"] is True
    assert tool_event.data["content"] == "6*7 = 42"
