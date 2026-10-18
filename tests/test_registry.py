import asyncio

import pytest
from conftest import Spy

from bitguide.agent import Agent
from bitguide.llm import OfflineLLM
from bitguide.registry import AgentRegistry


def make_agent(agent_id: str = "a") -> tuple[Agent, Spy]:
    spy = Spy()
    return Agent(agent_id, "你是助手", OfflineLLM(), resources=[spy]), spy


async def settle() -> None:
    """让后台的退役任务有机会跑完。"""
    for _ in range(5):
        await asyncio.sleep(0)


async def test_register_acquire_and_errors():
    registry = AgentRegistry()
    agent, _ = make_agent()
    await registry.register(agent)
    async with registry.acquire("a") as borrowed:
        assert borrowed is agent
    with pytest.raises(ValueError):
        await registry.register(make_agent()[0])
    with pytest.raises(KeyError):
        async with registry.acquire("missing"):
            pass


async def test_replace_keeps_old_agent_alive_until_in_flight_request_finishes():
    registry = AgentRegistry()
    old, old_spy = make_agent()
    await registry.register(old)

    async with registry.acquire("a") as in_flight:
        new, _ = make_agent()
        assert await registry.replace(new) == 2
        async with registry.acquire("a") as fresh:
            assert fresh is new  # 新请求立刻拿到新实例
        await settle()
        assert in_flight is old
        assert not old_spy.closed  # 旧请求还在用，不能关

    await settle()
    assert old_spy.closed  # 最后一个引用归还后才关闭


async def test_reference_is_released_when_request_fails():
    registry = AgentRegistry()
    old, old_spy = make_agent()
    await registry.register(old)
    with pytest.raises(RuntimeError):
        async with registry.acquire("a"):
            raise RuntimeError("boom")
    await registry.replace(make_agent()[0])
    await settle()
    assert old_spy.closed


async def test_remove_and_infos():
    registry = AgentRegistry()
    a, a_spy = make_agent("a")
    await registry.register(a)
    await registry.register(make_agent("b")[0])
    await registry.replace(make_agent("b")[0])
    assert [(i.id, i.version) for i in await registry.infos()] == [("a", 1), ("b", 2)]
    assert await registry.remove("a") is True
    assert await registry.remove("a") is False
    await settle()
    assert a_spy.closed
    assert await registry.ids() == ["b"]


async def test_close_force_closes_leaked_references():
    registry = AgentRegistry()
    agent, spy = make_agent()
    await registry.register(agent)
    leaked = registry.acquire("a")
    await leaked.__aenter__()  # 模拟忘了归还的引用
    await registry.close(grace_seconds=0.05)
    assert spy.closed
