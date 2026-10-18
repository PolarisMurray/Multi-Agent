"""Skills：用 Markdown 文件给 Agent 按需“插”一段专业 Prompt。

一个 Skill 就是 skills/<名字>/SKILL.md：
    ---
    name: parenting-activity-advice
    description: 根据孩子年龄推荐亲子活动
    triggers: [亲子活动, 周末, 玩什么]
    tools: [parenting_age_calculator]
    ---
    正文就是要追加到 System Prompt 里的专业指令。

每轮对话先匹配用户问题，只把命中的 Skill 拼进 Prompt，避免 Prompt 无限膨胀。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(slots=True, frozen=True)
class Skill:
    name: str
    description: str
    prompt: str
    triggers: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()

    def score(self, message: str) -> int:
        """命中几个触发词就得几分；0 分表示不相关。"""
        text = message.casefold()
        return sum(1 for word in self.triggers if word.casefold() in text)


def load_skill(path: str | Path) -> Skill:
    source = Path(path)
    match = _FRONTMATTER.match(source.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"SKILL.md 缺少 YAML frontmatter: {source}")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"frontmatter 必须是键值对: {source}")
    name = str(meta.get("name", "")).strip()
    description = str(meta.get("description", "")).strip()
    prompt = match.group(2).strip()
    if not (name and description and prompt):
        raise ValueError(f"SKILL.md 需要 name、description 和正文: {source}")
    return Skill(
        name=name,
        description=description,
        prompt=prompt,
        triggers=tuple(str(word) for word in meta.get("triggers", [])),
        tools=tuple(str(tool) for tool in meta.get("tools", [])),
    )


class SkillManager:
    def __init__(self, directory: str | Path, enabled: list[str] | None = None):
        """enabled 是白名单；None 表示目录里的 Skill 全部可用。"""
        self.directory = Path(directory)
        paths = sorted(self.directory.glob("*/SKILL.md")) if self.directory.exists() else []
        skills = [load_skill(path) for path in paths]
        self._skills = {s.name: s for s in skills if enabled is None or s.name in enabled}
        if enabled is not None:
            missing = set(enabled) - set(self._skills)
            if missing:
                # 配置里拼错 Skill 名要在启动时就报错，而不是运行时悄悄不生效
                raise ValueError(f"找不到 Skill: {', '.join(sorted(missing))}")
        # TODO(改写): 热重载——记录文件 mtime，变化时重新扫描目录

    def names(self) -> list[str]:
        return sorted(self._skills)

    def match(self, message: str) -> list[Skill]:
        # TODO(改写): 关键词匹配召回率有限，可以再加一层 LLM / Embedding 语义路由兜底
        scored = [(skill.score(message), skill) for skill in self._skills.values()]
        return [skill for score, skill in sorted(scored, key=lambda p: (-p[0], p[1].name)) if score]

    def enhance_prompt(self, base_prompt: str, message: str) -> tuple[str, list[Skill]]:
        matched = self.match(message)
        if not matched:
            return base_prompt, []
        sections = [base_prompt, "\n\n# 本轮启用的技能"]
        for skill in matched:
            tools = "、".join(skill.tools) or "无专用工具"
            sections.append(f"\n## 技能：{skill.name}\n可用工具：{tools}\n{skill.prompt}")
        return "".join(sections), matched
