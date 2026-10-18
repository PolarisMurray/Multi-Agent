import pytest
from conftest import CONFIG, ROOT
from pydantic import ValidationError

from bitguide.config import PlatformSettings


def minimal(**agent_fields) -> dict:
    agent = {"id": "a", "name": "A", "modelRef": "m", "systemPrompt": "hi", **agent_fields}
    return {"models": {"m": {"provider": "offline"}}, "agents": {"a": agent}}


def test_project_config_loads_and_resolves_paths():
    settings = PlatformSettings.load(CONFIG)
    assert settings.base_dir == ROOT
    assert list(settings.agents) == ["parenting_advisor", "tech_helper", "content_editor"]
    assert settings.resolve(settings.knowledge_bases["parenting"]) == ROOT / "knowledge/parenting"
    assert settings.agents["parenting_advisor"].mcp_servers == ["parenting"]


def test_camel_case_and_snake_case_are_both_accepted():
    settings = PlatformSettings.model_validate(
        {"models": {"m": {"modelName": "x", "max_tokens": 10}}}
    )
    assert settings.models["m"].model_name == "x"
    assert settings.models["m"].max_tokens == 10


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"modelRef": "missing"}, "unknown model"),
        ({"knowledgeBase": "kb"}, "unknown knowledge base"),
        ({"mcpServers": ["srv"]}, "unknown MCP server"),
    ],
)
def test_references_are_checked(fields, message):
    with pytest.raises(ValidationError, match=message):
        PlatformSettings.model_validate(minimal(**fields))


def test_typo_in_yaml_key_is_rejected():
    with pytest.raises(ValidationError, match="maxIteration"):
        PlatformSettings.model_validate(minimal(maxIteration=3))


def test_agent_key_must_equal_id():
    raw = minimal()
    raw["agents"] = {"b": raw["agents"]["a"]}
    with pytest.raises(ValidationError, match="must equal id"):
        PlatformSettings.model_validate(raw)


def test_splitter_overlap_must_be_smaller_than_size():
    raw = minimal()
    raw["rag"] = {"splitter": {"maxSegmentSize": 100, "overlap": 100}}
    with pytest.raises(ValidationError, match="overlap"):
        PlatformSettings.model_validate(raw)


def test_offline_switches_models_and_vectors_without_touching_original():
    original = PlatformSettings.load(CONFIG)
    offline = original.offline()
    assert {m.provider for m in offline.models.values()} == {"offline"}
    assert offline.rag.embedding.provider == "hash"
    assert offline.rag.vector_store.type == "memory"
    assert original.models["local"].provider == "ollama"


def test_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret")
    settings = PlatformSettings.model_validate(
        {"models": {"m": {"apiKeyEnv": "TEST_API_KEY", "apiKey": "ignored"}}}
    )
    assert settings.models["m"].resolved_api_key() == "secret"


def test_etcd_prefix_gets_a_trailing_slash():
    settings = PlatformSettings.model_validate({"models": {}, "etcd": {"prefix": "/x"}})
    assert settings.etcd.prefix == "/x/"
