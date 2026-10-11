import asyncio

import pytest

from bitguide.memory import ConversationMemory, Interaction, MemoryStore


async def test_add_turn_and_messages():
    m = ConversationMemory()
    await m.add_turn("s1", "你好", "你好！有什么可以帮你？")
    msgs = await m.messages("s1")
    assert [(x.role, x.content) for x in msgs] == [
        ("user", "你好"),
        ("assistant", "你好！有什么可以帮你？"),
    ]


async def test_tool_interaction_keeps_call_id():
    m = ConversationMemory()
    await m.add_tool("s1", "calculator", "3 * 7 = 21", "call_1")
    (msg,) = await m.messages("s1")
    assert (msg.role, msg.name, msg.tool_call_id) == ("tool", "calculator", "call_1")


async def test_sessions_are_isolated():
    m = ConversationMemory()
    await m.add_turn("alice", "a", "b")
    await m.add_turn("bob", "c", "d")
    assert [x.content for x in await m.history("alice")] == ["a", "b"]
    assert await m.session_count() == 2


async def test_trims_oldest_first():
    m = ConversationMemory(max_history_length=3)
    for i in range(5):
        await m.add("s1", Interaction("user", str(i)))
    assert [x.content for x in await m.history("s1")] == ["2", "3", "4"]


def test_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        ConversationMemory(max_history_length=0)


async def test_history_returns_copy():
    m = ConversationMemory()
    await m.add_turn("s1", "a", "b")
    (await m.history("s1")).clear()
    assert len(await m.history("s1")) == 2


async def test_reading_unknown_session_does_not_create_it():
    m = ConversationMemory()
    assert await m.history("ghost") == []
    assert await m.messages("ghost") == []
    assert await m.session_count() == 0  # 坑 ④


async def test_clear_is_idempotent():
    m = ConversationMemory()
    await m.add_turn("s1", "a", "b")
    await m.clear("s1")
    await m.clear("s1")  # 再清一次也不能报错
    assert await m.history("s1") == []


async def test_concurrent_adds_are_not_lost():
    m = ConversationMemory(max_history_length=1000)
    await asyncio.gather(*(m.add("s1", Interaction("user", str(i))) for i in range(200)))
    assert len(await m.history("s1")) == 200


def test_timestamp_is_timezone_aware():
    assert Interaction("user", "x").timestamp.tzinfo is not None


def test_conversation_memory_satisfies_protocol():
    store: MemoryStore = ConversationMemory()
    assert store is not None
