import pytest
from conftest import ROOT, ScriptedLLM

from bitguide.agent import Agent
from bitguide.models import AgentRequest, ModelTurn
from bitguide.observability import RecordingTracer
from bitguide.skills import SkillManager, load_skill


def write_skill(root, name: str, triggers: list[str], body: str = "正文指令") -> None:
    folder = root / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} 的描述\ntriggers: {triggers}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_project_skill_is_parsed():
    skill = load_skill(ROOT / "skills/parenting-activity-advice/SKILL.md")
    assert skill.name == "parenting-activity-advice"
    assert "亲子活动" in skill.triggers
    assert "parenting_age_calculator" in skill.tools
    assert skill.prompt.startswith("你现在是")


def test_skill_without_frontmatter_is_rejected(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("只有正文", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter"):
        load_skill(path)


def test_match_orders_by_score(tmp_path):
    write_skill(tmp_path, "weekend", ["周末", "孩子"])
    write_skill(tmp_path, "trip", ["周末"])
    manager = SkillManager(tmp_path)
    assert [s.name for s in manager.match("周末带孩子去哪")] == ["weekend", "trip"]
    assert manager.match("今天天气") == []


def test_whitelist_and_typo_detection(tmp_path):
    write_skill(tmp_path, "a", ["x"])
    write_skill(tmp_path, "b", ["y"])
    assert SkillManager(tmp_path, enabled=["a"]).names() == ["a"]
    with pytest.raises(ValueError, match="zzz"):
        SkillManager(tmp_path, enabled=["zzz"])


def test_enhance_prompt_only_when_matched(tmp_path):
    write_skill(tmp_path, "a", ["周末"], body="请推荐活动")
    manager = SkillManager(tmp_path)
    prompt, matched = manager.enhance_prompt("基础指令", "周末玩什么")
    assert prompt.startswith("基础指令")
    assert "## 技能：a" in prompt and "请推荐活动" in prompt
    assert [s.name for s in matched] == ["a"]
    assert manager.enhance_prompt("基础指令", "无关") == ("基础指令", [])


async def test_agent_injects_matched_skill_into_system_prompt(tmp_path):
    write_skill(tmp_path, "a", ["周末"], body="请推荐活动")
    llm = ScriptedLLM(ModelTurn(content="好的"))
    tracer = RecordingTracer()
    agent = Agent("x", "你是助手", llm, skills=SkillManager(tmp_path), tracer=tracer)
    await agent.run(AgentRequest("周末玩什么", "s"))
    system = llm.calls[0][0]
    assert system.role == "system" and "请推荐活动" in system.content
    assert "skills.matched" in tracer.kinds()
