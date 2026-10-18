import pytest
from conftest import Spy

from bitguide import factory as factory_module
from bitguide.factory import AgentFactory
from bitguide.llm import OfflineLLM, OpenAICompatibleClient
from bitguide.models import AgentRequest
from bitguide.observability import RecordingTracer


async def test_builds_a_complete_agent_from_config(offline_settings):
    tracer = RecordingTracer()
    agent = await AgentFactory(offline_settings, tracer).build(
        offline_settings.agents["parenting_advisor"]
    )
    try:
        assert agent.name == "亲子教育顾问"
        assert {"calculator", "datetime", "parenting_age_calculator"} <= set(agent.tool_names())
        response = await agent.run(AgentRequest("周末带孩子玩什么亲子活动", "s"))
        assert "parenting-activity-advice" in response.content  # Skill 生效
        assert "参考资料" in response.content  # RAG 生效
    finally:
        await agent.close()


def test_model_provider_decides_the_client(offline_settings):
    factory = AgentFactory(offline_settings)
    assert isinstance(factory.build_llm("local"), OfflineLLM)
    offline_settings.models["local"].provider = "ollama"
    assert isinstance(factory.build_llm("local"), OpenAICompatibleClient)


async def test_failed_assembly_closes_resources_already_created(offline_settings, monkeypatch):
    spy = Spy()

    async def fake_connect(*args, **kwargs):
        return spy, []

    monkeypatch.setattr(factory_module, "connect_stdio_server", fake_connect)
    # MCP 先连上，随后 Skill 名字写错导致失败 → 已经起来的 MCP 必须被关掉
    spec = offline_settings.agents["parenting_advisor"].model_copy(update={"skills": ["nope"]})
    with pytest.raises(ValueError, match="nope"):
        await AgentFactory(offline_settings).build(spec)
    assert spy.closed


async def test_broken_mcp_server_degrades_instead_of_failing(offline_settings):
    offline_settings.mcp_servers["parenting"].command = "/definitely/not/a/program"
    tracer = RecordingTracer()
    agent = await AgentFactory(offline_settings, tracer).build(
        offline_settings.agents["parenting_advisor"]
    )
    assert "parenting_age_calculator" not in agent.tool_names()
    assert "agent.degraded" in tracer.kinds()
    await agent.close()


class BrokenEmbedder:
    async def embed(self, texts):
        raise RuntimeError("embedding down")


async def test_rag_failure_respects_fail_open(offline_settings, monkeypatch):
    monkeypatch.setattr(AgentFactory, "_build_embedder", lambda self: BrokenEmbedder())
    spec = offline_settings.agents["tech_helper"]
    tracer = RecordingTracer()
    agent = await AgentFactory(offline_settings, tracer).build(spec)  # fail-open：照样能用
    assert "agent.degraded" in tracer.kinds()
    await agent.close()

    offline_settings.rag.fail_open = False
    with pytest.raises(RuntimeError, match="embedding down"):
        await AgentFactory(offline_settings).build(spec)
