"""LLM 适配层：整个项目里只有这个文件认识 OpenAI 的线上格式。

- OpenAICompatibleClient：对接 Ollama / OpenAI / 任何 OpenAI 兼容服务。
- OfflineLLM：不联网的规则模型，让项目在没有大模型时也能完整跑通 ReAct 和工作流。
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from .models import Message, ModelTurn, ToolCall


def to_openai_message(message: Message) -> dict[str, Any]:
    """把内部 Message 翻译成 OpenAI 的 JSON；值为 None 的字段一律不发。"""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # arguments 在线上是 JSON 字符串；ensure_ascii=False 保留中文原样
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    """解析模型给的工具参数。

    小模型偶尔会输出不合法的 JSON。这里不直接抛异常，而是把原文交给工具，
    工具校验失败后会返回错误信息给模型，模型通常会在下一轮自己修正。
    TODO(改写): 也可以选择直接失败或重试请求，想想三种策略各自的代价。
    """
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_invalid_json": raw}
    return value if isinstance(value, dict) else {"_invalid_json": raw}


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        # SDK 不接受空 key；Ollama 根本不校验 key，所以给个占位值
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "ollama",
            timeout=timeout,
            # openai 3.x 的类型标注写的是 httpx2，但运行时同样接受 httpx.AsyncClient
            http_client=http_client,  # type: ignore[arg-type]
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[to_openai_message(m) for m in messages],  # type: ignore[misc]
            # 没有工具时整个字段都不发：传 None 会变成 "tools": null，有的服务端会拒绝
            tools=tools if tools else openai.omit,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message
        calls = [
            ToolCall(call.id, call.function.name, parse_tool_arguments(call.function.arguments))
            for call in message.tool_calls or []
            if call.type == "function"
        ]
        # 模型决定调工具时 content 通常是 None
        return ModelTurn(content=message.content or "", tool_calls=calls)

    async def close(self) -> None:
        await self._client.close()


# 至少包含一个运算符的算式，例如 "12*(3+4)"
_EXPRESSION = re.compile(r"[\d.(][\d.\s()]*(?:[-+*/%][\d.\s()]+)+")
_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")


class OfflineLLM:
    """确定性的“假模型”，只用来演示和测试流程，不追求回答质量。

    规则：
    1. 上一条是工具结果 → 根据结果作答（ReAct 的 Observation → Answer）。
    2. 问题里有算式且有 calculator 工具 → 发起工具调用（Reason → Act）。
    3. 其他情况 → 用“角色 + 问题 + 参考资料 + 启用技能”拼一个回答。
    """

    def __init__(self) -> None:
        self._ids = itertools.count(1)

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn:
        last = messages[-1]
        if last.role == "tool":
            return ModelTurn(content=f"根据工具 {last.name} 的结果：{last.content}")

        question = last.content
        tool_names = {tool["function"]["name"] for tool in tools}
        match = _EXPRESSION.search(question)
        if "calculator" in tool_names and match and not _DATE.search(question):
            call = ToolCall(
                f"call_{next(self._ids)}", "calculator", {"expression": match.group().strip()}
            )
            return ModelTurn(tool_calls=[call])

        system = messages[0].content if messages and messages[0].role == "system" else ""
        lines = [f"[离线模型·{_role_of(system)}] 关于「{_shorten(question, 80)}」的回答。"]
        if reference := _first_reference(system):
            lines.append(f"参考资料：{_shorten(reference, 60)}")
        if skills := re.findall(r"^## 技能：(\S+)", system, flags=re.MULTILINE):
            lines.append(f"启用技能：{'、'.join(skills)}")
        return ModelTurn(content="\n".join(lines))


def _role_of(system_prompt: str) -> str:
    first_sentence = re.split(r"[。.!！\n]", system_prompt.strip(), maxsplit=1)[0]
    return _shorten(first_sentence.removeprefix("你是"), 20) or "助手"


def _first_reference(system_prompt: str) -> str | None:
    _, found, tail = system_prompt.partition("参考资料：\n")
    if not found:
        return None
    first = tail.strip().splitlines()[0] if tail.strip() else ""
    return first.removeprefix("- ") or None


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
