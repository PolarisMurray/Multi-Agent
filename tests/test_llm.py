import json

import httpx

from bitguide.llm import OfflineLLM, OpenAICompatibleClient, parse_tool_arguments, to_openai_message
from bitguide.models import Message, ToolCall

CALCULATOR = [{"type": "function", "function": {"name": "calculator", "parameters": {}}}]


def make_client(reply: dict):
    """用 MockTransport 拦截 SDK 真正发出的 HTTP 请求，返回一个固定的响应。"""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": reply}],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        "http://test/v1", "k", "m", temperature=0.2, max_tokens=99, http_client=http
    )
    return client, sent


def test_plain_message_omits_optional_fields():
    assert to_openai_message(Message("user", "hi")) == {"role": "user", "content": "hi"}


def test_tool_call_arguments_are_a_json_string_keeping_chinese():
    message = Message("assistant", "", tool_calls=[ToolCall("c1", "weather", {"city": "北京"})])
    arguments = to_openai_message(message)["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert "北京" in arguments
    assert json.loads(arguments) == {"city": "北京"}


def test_tool_message_keeps_call_id_and_name():
    payload = to_openai_message(Message("tool", "84", name="calculator", tool_call_id="c1"))
    assert payload == {"role": "tool", "content": "84", "name": "calculator", "tool_call_id": "c1"}


async def test_no_tools_means_no_tools_field():
    client, sent = make_client({"role": "assistant", "content": "你好"})
    turn = await client.complete([Message("user", "hi")], [])
    await client.close()
    assert turn.content == "你好"
    assert turn.tool_calls == []
    assert "tools" not in sent[0]
    assert (sent[0]["model"], sent[0]["temperature"], sent[0]["max_tokens"]) == ("m", 0.2, 99)


async def test_tools_are_sent_and_tool_calls_parsed():
    reply = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
            }
        ],
    }
    client, sent = make_client(reply)
    turn = await client.complete([Message("user", "1+1")], CALCULATOR)
    await client.close()
    assert sent[0]["tools"] == CALCULATOR
    assert turn.content == ""
    assert turn.tool_calls == [ToolCall("c1", "calculator", {"expression": "1+1"})]


def test_invalid_json_arguments_are_passed_to_the_tool():
    assert parse_tool_arguments("{oops") == {"_invalid_json": "{oops"}
    assert parse_tool_arguments("[1]") == {"_invalid_json": "[1]"}
    assert parse_tool_arguments(None) == {}


async def test_offline_llm_calls_calculator_for_expressions():
    messages = [Message("system", "你是助手。"), Message("user", "算一下 12*(3+4) ")]
    turn = await OfflineLLM().complete(messages, CALCULATOR)
    assert turn.tool_calls[0].name == "calculator"
    assert turn.tool_calls[0].arguments == {"expression": "12*(3+4)"}


async def test_offline_llm_answers_from_tool_result():
    messages = [Message("user", "x"), Message("tool", "84", name="calculator", tool_call_id="c1")]
    turn = await OfflineLLM().complete(messages, [])
    assert "84" in turn.content
    assert not turn.tool_calls


async def test_offline_llm_shows_role_reference_and_skills():
    system = (
        "你是亲子顾问。\n\n# 本轮启用的技能\n## 技能：play\n正文\n\n参考资料：\n- [a.md] 多陪伴"
    )
    turn = await OfflineLLM().complete(
        [Message("system", system), Message("user", "2024-10-17 带孩子去哪")], CALCULATOR
    )
    assert not turn.tool_calls  # 日期不是算式
    assert "亲子顾问" in turn.content
    assert "[a.md] 多陪伴" in turn.content
    assert "play" in turn.content
