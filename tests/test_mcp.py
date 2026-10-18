import sys

import pytest
from conftest import ROOT, ScriptedLLM

from bitguide.agent import Agent
from bitguide.mcp import McpError, McpStdioClient, connect_stdio_server
from bitguide.models import AgentRequest, ModelTurn, ToolCall

SERVER = [sys.executable, str(ROOT / "mcp_servers" / "parenting_server.py")]


async def test_handshake_list_and_call_tools():
    client, tools = await connect_stdio_server("parenting", SERVER)
    try:
        assert client.server_info["name"] == "parenting"
        assert sorted(t.name for t in tools) == [
            "parenting_activity_ideas",
            "parenting_age_calculator",
        ]
        ideas = next(t for t in tools if t.remote_name == "activity_ideas")
        assert ideas.parameters["required"] == ["age"]

        ok = await ideas.invoke({"age": 5, "place": "outdoor"})
        assert ok.success and "户外" in ok.content
        bad_place = await ideas.invoke({"age": 5, "place": "moon"})
        assert not bad_place.success
        missing_arg = await ideas.invoke({})
        assert not missing_arg.success and "参数错误" in (missing_arg.error or "")
    finally:
        await client.close()


async def test_unknown_method_is_a_json_rpc_error():
    client, _ = await connect_stdio_server("parenting", SERVER)
    try:
        with pytest.raises(McpError, match="Method not found"):
            await client.request("resources/list", {})
    finally:
        await client.close()


async def test_server_that_exits_immediately():
    client = McpStdioClient([sys.executable, "-c", "pass"], timeout=5)
    with pytest.raises(McpError):
        await client.start()


async def test_request_timeout():
    silent = McpStdioClient([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.3)
    with pytest.raises(McpError, match="timed out"):
        await silent.start()


async def test_agent_calls_an_mcp_tool_like_a_local_tool():
    client, tools = await connect_stdio_server("parenting", SERVER)
    llm = ScriptedLLM(
        ModelTurn(tool_calls=[ToolCall("c1", "parenting_age_calculator", {"birth_year": 2021})]),
        ModelTurn(content="完成"),
    )
    agent = Agent("p", "你是顾问", llm, tools, resources=[client])
    try:
        await agent.run(AgentRequest("孩子几岁", "s"))
        tool_message = llm.calls[1][-1]
        assert tool_message.role == "tool"
        assert "周岁" in tool_message.content
    finally:
        await agent.close()
