"""一个手写的最小 MCP Server（stdio 传输），不依赖任何 SDK。

运行方式：由 Agent 以子进程启动，通过 stdin/stdout 逐行交换 JSON-RPC 消息。
手动调试（在项目根目录执行）：
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python mcp_servers/parenting_server.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "age_calculator",
        "description": "根据出生年份计算孩子的周岁年龄和所处发展阶段",
        "inputSchema": {
            "type": "object",
            "properties": {"birth_year": {"type": "integer", "description": "出生年份，如 2021"}},
            "required": ["birth_year"],
        },
    },
    {
        "name": "activity_ideas",
        "description": "按年龄和场地（indoor/outdoor）给出亲子活动建议",
        "inputSchema": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 12},
                "place": {"type": "string", "enum": ["indoor", "outdoor"]},
            },
            "required": ["age"],
        },
    },
]

STAGES = [(3, "婴幼儿期"), (6, "学龄前"), (12, "学龄期"), (200, "青少年及以上")]

IDEAS = {
    ("indoor", 0): "感官游戏、亲子共读绘本、搭软积木",
    ("indoor", 4): "拼图、角色扮演、简单手工",
    ("indoor", 7): "桌游、科学小实验、编程启蒙积木",
    ("outdoor", 0): "公园散步、晒太阳、沙坑玩耍",
    ("outdoor", 4): "骑平衡车、捡树叶做标本、野餐",
    ("outdoor", 7): "骑自行车、徒步、观鸟记录",
}


def text(content: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": content}], "isError": error}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "age_calculator":
        age = date.today().year - int(arguments["birth_year"])
        if not 0 <= age <= 120:
            return text("出生年份不合法", error=True)
        stage = next(label for limit, label in STAGES if age < limit)
        return text(f"孩子约 {age} 周岁，处于{stage}。")
    if name == "activity_ideas":
        age = int(arguments["age"])
        place = str(arguments.get("place", "indoor"))
        if place not in ("indoor", "outdoor"):
            return text("place 只能是 indoor 或 outdoor", error=True)
        bucket = 7 if age >= 7 else 4 if age >= 4 else 0
        return text(
            f"{age} 岁{'室内' if place == 'indoor' else '户外'}活动：{IDEAS[(place, bucket)]}"
        )
    return text(f"未知工具：{name}", error=True)


def handle(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "parenting", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            result = call_tool(str(params.get("name")), dict(params.get("arguments") or {}))
        except (KeyError, TypeError, ValueError) as error:
            result = text(f"参数错误：{error}", error=True)
    else:
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {"code": -32601, "message": "Method not found"},
        }
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if "id" not in request:  # 通知（例如 notifications/initialized）不需要回复
            continue
        print(json.dumps(handle(request), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
