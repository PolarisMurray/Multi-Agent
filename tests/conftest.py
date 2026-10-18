from pathlib import Path

import pytest

from bitguide.config import PlatformSettings
from bitguide.models import Message, ModelTurn

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "platform.yaml"


class ScriptedLLM:
    """按顺序返回预先写好的 ModelTurn，并记录每次收到的消息。"""

    def __init__(self, *turns: ModelTurn):
        self._turns = list(turns)
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools):
        self.calls.append(list(messages))
        return self._turns.pop(0)


class Spy:
    """一个假资源：记录自己有没有被 close。"""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def offline_settings(tmp_path) -> PlatformSettings:
    """项目真实配置的离线版本；Checkpoint 写到临时目录，不弄脏仓库。"""
    settings = PlatformSettings.load(CONFIG).offline()
    settings.workflow.checkpoint.directory = tmp_path / "checkpoints"
    return settings
