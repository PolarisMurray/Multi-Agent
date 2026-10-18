"""MCP（Model Context Protocol）客户端：JSON-RPC 2.0 over stdio。

协议流程（一行一条 JSON 消息）：
    client → initialize                    （握手，交换协议版本和能力）
    client → notifications/initialized     （通知，没有 id，不需要回复）
    client → tools/list                    （拿到工具清单和参数 JSON Schema）
    client → tools/call {name, arguments}  （真正调用工具）

McpTool 把远端工具包装成和本地工具一样的 Tool 协议，Agent 完全感知不到区别。
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .tools import ToolResult

PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    pass


class McpStdioClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30,
    ):
        if not command:
            raise ValueError("MCP command cannot be empty")
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env or {})
        self.timeout = timeout
        self.server_info: dict[str, Any] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        # 一条 stdio 管道上同一时间只能有一个“请求-响应”在进行，否则响应会对不上号
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.cwd,
            env={**os.environ, **self.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,  # TODO(改写): 收集 stderr 方便排查 Server 报错
        )
        try:
            result = await self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "bitguide", "version": "0.1.0"},
                },
            )
            self.server_info = dict(result.get("serverInfo", {}))
            await self.notify("notifications/initialized", {})
        except Exception:
            await self.close()
            raise

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise McpError("MCP server is not running")
        try:
            process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as error:
            raise McpError("MCP server exited unexpectedly") from error

    async def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            assert self._process is not None and self._process.stdout is not None
            while True:
                try:
                    line = await asyncio.wait_for(self._process.stdout.readline(), self.timeout)
                except TimeoutError as error:
                    raise McpError(f"MCP request timed out: {method}") from error
                if not line:
                    raise McpError("MCP server closed the connection")
                message = json.loads(line)
                # 跳过 Server 主动发来的通知（没有 id）或其他请求的响应
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise McpError(str(message["error"].get("message", message["error"])))
                return dict(message.get("result") or {})

    async def notify(self, method: str, params: Mapping[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def list_tools(self) -> list[dict[str, Any]]:
        return list((await self.request("tools/list", {})).get("tools", []))

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return await self.request("tools/call", {"name": name, "arguments": dict(arguments)})

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 3)
        except TimeoutError:
            process.kill()
            await process.wait()


class McpTool:
    """把 MCP Server 上的一个工具适配成本地 Tool 协议（适配器模式）。"""

    def __init__(self, server: str, client: McpStdioClient, spec: Mapping[str, Any]):
        self.remote_name = str(spec["name"])
        # 加 server 前缀，避免两个 Server 都有同名工具时冲突
        self.name = f"{server}_{self.remote_name}"
        self.description = str(spec.get("description", ""))
        self.parameters = dict(spec.get("inputSchema") or {"type": "object", "properties": {}})
        self._client = client

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        result = await self._client.call_tool(self.remote_name, arguments)
        text = "\n".join(
            str(item.get("text", ""))
            for item in result.get("content", [])
            if item.get("type") == "text"
        )
        return (
            ToolResult.fail(text or "MCP tool error")
            if result.get("isError")
            else ToolResult.ok(text)
        )


async def connect_stdio_server(
    server: str,
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 30,
) -> tuple[McpStdioClient, list[McpTool]]:
    """启动子进程、握手、拉取工具清单。返回 client（调用方负责 close）和工具列表。"""
    client = McpStdioClient(command, cwd=cwd, env=env, timeout=timeout_seconds)
    await client.start()
    try:
        tools = [McpTool(server, client, spec) for spec in await client.list_tools()]
    except Exception:
        await client.close()
        raise
    return client, tools
