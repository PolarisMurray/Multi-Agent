import json

import pytest
import yaml
from conftest import CONFIG, ROOT

from bitguide.cli import main


@pytest.fixture
def config(tmp_path):
    """项目配置的副本：baseDir 指回项目根，Checkpoint 写到临时目录。"""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["baseDir"] = str(ROOT)
    raw["workflow"]["checkpoint"]["directory"] = str(tmp_path / "checkpoints")
    path = tmp_path / "platform.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def cli(config, capsys):
    def run(*args: str) -> tuple[int, str, str]:
        code = main(["-c", str(config), "--offline", "--quiet", *args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


def test_agents_command(cli):
    code, out, _ = cli("agents")
    assert code == 0
    assert "parenting_advisor" in out and "parenting_age_calculator" in out


def test_chat_one_shot_uses_tools(cli):
    code, out, _ = cli("chat", "--agent", "tech_helper", "-m", "算一下 2*21")
    assert code == 0 and "42" in out


def test_chat_defaults_to_first_configured_agent(cli):
    code, out, _ = cli("chat", "-m", "你好")
    assert code == 0 and "亲子教育顾问" in out


def test_workflow_list(cli):
    code, out, _ = cli("workflow", "list")
    assert code == 0 and len(out.strip().splitlines()) == 8


def test_workflow_pause_and_resume_across_invocations(cli):
    code, out, _ = cli("workflow", "run", "human_approval", "-i", "周五团建", "--thread", "t1")
    assert code == 0 and "interrupted" in out and "resume t1 --approve" in out

    code, _, err = cli("workflow", "resume", "t1")
    assert code == 1 and "--approve" in err  # 等审批时必须给出决定

    code, out, _ = cli("workflow", "resume", "t1", "--approve")
    assert code == 0 and "completed" in out and "已发布" in out

    code, out, _ = cli("workflow", "show", "t1")
    assert json.loads(out)["status"] == "completed"

    code, out, _ = cli("workflow", "history")
    assert "t1" in out and "human_approval" in out


def test_errors_are_reported_with_exit_code(cli):
    code, _, err = cli("workflow", "run", "nope", "-i", "x")
    assert code == 1 and "workflow not found" in err
    code, _, err = cli("workflow", "show", "missing")
    assert code == 1
