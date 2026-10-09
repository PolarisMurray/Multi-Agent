import pytest

from bitguide.models import AgentRequest, AgentResponse, Message, ModelTurn, ToolCall


def test_message_defaults():
    msg = Message(role="user", content="Hello")
    assert msg.name is None
    assert msg.tool_call_id is None
    assert msg.tool_calls == []


def test_tool_call_creation():
    a = Message(role="assistant", content="")
    b = Message(role="assistant", content="")
    a.tool_calls.append(ToolCall(id="1", name="tool1", arguments={"arg1": "value1"}))
    assert b.tool_calls == []


def test_slots_reject_unknown_attributes():
    msg = Message(role="user", content="Hello")
    with pytest.raises(AttributeError):
        msg.contnet = "typo"  # type: ignore


def test_tool_round_trip_shape():
    call = ToolCall(id="c1", name="calculator", arguments={"expression": "3 * 7"})
    turn = ModelTurn(tool_calls=[call])
    assert turn.tool_calls[0].id == "c1"
    reply = Message(role="tool", content="21", name=call.name, tool_call_id=call.id)
    assert turn.content == ""
    assert reply.tool_call_id == turn.tool_calls[0].id


def test_request_and_response():
    req = AgentRequest(message="Hello", session_id="s1")
    assert req.request_id is None
    resp = AgentResponse(content="Hi", agent_id="parenting_advisor", session_id="s1")
    assert resp.iterations == 1
