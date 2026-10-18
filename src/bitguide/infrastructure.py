"""etcd v3 客户端（走 etcd 自带的 HTTP/JSON 网关，不需要 gRPC 依赖）。

etcd 网关的两个约定：
1. key / value 都要 base64 编码后放进 JSON。
2. 前缀查询用 [key, range_end) 区间表示：range_end = 前缀最后一个字节 + 1。
   例如 "/agents/" 的 range_end 是 "/agents0"（'/' 的下一个字符是 '0'）。

Watch 是一个长连接：服务端每发生一次变更就推送一行 JSON，客户端逐行读取。
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import httpx


def _b64(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.b64encode(raw).decode()


def _unb64(value: str | None) -> str | None:
    return None if value is None else base64.b64decode(value).decode()


def prefix_range_end(prefix: str) -> bytes:
    """返回“比所有以 prefix 开头的 key 都大”的最小 key。"""
    raw = bytearray(prefix.encode())
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] < 0xFF:
            raw[i] += 1
            return bytes(raw[: i + 1])
    return b"\0"  # 前缀全是 0xFF：etcd 约定用 \0 表示“直到末尾”


@dataclass(slots=True, frozen=True)
class WatchEvent:
    kind: Literal["PUT", "DELETE"]
    key: str
    value: str | None
    revision: int


class EtcdClient:
    def __init__(
        self,
        endpoint: str = "http://localhost:2379",
        timeout: float = 5,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._http = http_client or httpx.AsyncClient(
            base_url=endpoint.rstrip("/"), timeout=timeout
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._http.post(path, json=body)
        response.raise_for_status()
        return response.json()

    async def put(self, key: str, value: str) -> None:
        await self._post("/v3/kv/put", {"key": _b64(key), "value": _b64(value)})

    async def get(self, key: str) -> str | None:
        data = await self._post("/v3/kv/range", {"key": _b64(key)})
        kvs = data.get("kvs", [])
        return _unb64(kvs[0].get("value", "")) if kvs else None

    async def get_prefix(self, prefix: str) -> dict[str, str]:
        body = {"key": _b64(prefix), "range_end": _b64(prefix_range_end(prefix))}
        data = await self._post("/v3/kv/range", body)
        return {
            _unb64(kv["key"]) or "": _unb64(kv.get("value", "")) or "" for kv in data.get("kvs", [])
        }

    async def delete(self, key: str) -> bool:
        data = await self._post("/v3/kv/deleterange", {"key": _b64(key)})
        return int(data.get("deleted", 0)) > 0

    async def watch(
        self, prefix: str, start_revision: int | None = None
    ) -> AsyncIterator[WatchEvent]:
        """持续产出 prefix 下的变更事件，直到调用方停止迭代。

        TODO(改写): 断线重连——记住最后的 revision，重连时从 revision + 1 继续，
        才不会漏掉断线期间的变更（这就是第 15 课“热更新”的基础）。
        """
        request: dict[str, Any] = {"key": _b64(prefix), "range_end": _b64(prefix_range_end(prefix))}
        if start_revision is not None:
            request["start_revision"] = start_revision
        async with self._http.stream(
            "POST", "/v3/watch", json={"create_request": request}, timeout=None
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                result = json.loads(line).get("result", {})
                for event in result.get("events", []):
                    kv = event.get("kv", {})
                    # protobuf 的枚举默认值（PUT=0）在 JSON 里会被省略，所以缺省就是 PUT
                    kind = "DELETE" if event.get("type") == "DELETE" else "PUT"
                    yield WatchEvent(
                        kind=kind,
                        key=_unb64(kv.get("key")) or "",
                        value=_unb64(kv.get("value")) if kind == "PUT" else None,
                        revision=int(kv.get("mod_revision", 0)),
                    )

    async def close(self) -> None:
        await self._http.aclose()
