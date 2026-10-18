import base64
import json

import httpx

from bitguide.application import load_agents_from_etcd, seed_agents_to_etcd
from bitguide.config import AgentSettings
from bitguide.infrastructure import EtcdClient, WatchEvent, prefix_range_end


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode()


def unb64(value: str) -> bytes:
    return base64.b64decode(value)


def fake_etcd() -> EtcdClient:
    """一个只实现了 4 个接口的 etcd JSON 网关，用来测试客户端的编码和区间逻辑。"""
    data: dict[bytes, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        path = request.url.path
        if path == "/v3/kv/put":
            data[unb64(body["key"])] = unb64(body["value"])
            return httpx.Response(200, json={})
        if path == "/v3/kv/range":
            key = unb64(body["key"])
            end = unb64(body["range_end"]) if "range_end" in body else None
            keys = [k for k in sorted(data) if (k == key if end is None else key <= k < end)]
            kvs = [{"key": b64(k), "value": b64(data[k])} for k in keys]
            return httpx.Response(200, json={"kvs": kvs} if kvs else {})
        if path == "/v3/kv/deleterange":
            existed = data.pop(unb64(body["key"]), None) is not None
            return httpx.Response(200, json={"deleted": str(int(existed))})
        if path == "/v3/watch":
            events = [
                {"kv": {"key": b64(b"/p/a"), "value": b64(b"v1"), "mod_revision": "7"}},
                {"type": "DELETE", "kv": {"key": b64(b"/p/b"), "mod_revision": "8"}},
            ]
            lines = [{"result": {"created": True}}, {"result": {"events": events}}]
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        return httpx.Response(404)

    http = httpx.AsyncClient(base_url="http://etcd", transport=httpx.MockTransport(handler))
    return EtcdClient(http_client=http)


def test_prefix_range_end():
    assert prefix_range_end("/agents/") == b"/agents0"
    assert prefix_range_end("abc") == b"abd"


async def test_put_get_prefix_and_delete():
    etcd = fake_etcd()
    await etcd.put("/p/a", "1")
    await etcd.put("/p/b", "中文")
    await etcd.put("/q/c", "3")
    assert await etcd.get("/p/b") == "中文"
    assert await etcd.get("/missing") is None
    assert await etcd.get_prefix("/p/") == {"/p/a": "1", "/p/b": "中文"}
    assert await etcd.delete("/p/a") is True
    assert await etcd.delete("/p/a") is False
    await etcd.close()


async def test_watch_parses_put_and_delete_events():
    etcd = fake_etcd()
    events = [event async for event in etcd.watch("/p/")]
    assert events == [
        WatchEvent("PUT", "/p/a", "v1", 7),
        WatchEvent("DELETE", "/p/b", None, 8),
    ]
    await etcd.close()


async def test_agent_definitions_round_trip_through_etcd():
    etcd = fake_etcd()
    spec = AgentSettings(id="a", name="A", model_ref="m", system_prompt="hi", tools=["calculator"])
    await seed_agents_to_etcd(etcd, "/bitguide/agents/", [spec])
    raw = await etcd.get("/bitguide/agents/a")
    assert raw is not None and '"modelRef"' in raw  # 存进去的是 camelCase，和 YAML 一致
    assert await load_agents_from_etcd(etcd, "/bitguide/agents/") == {"a": spec}
    await etcd.close()
