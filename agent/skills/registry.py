"""
SkillRegistry - Skill 的注册与发现

仿照 LLMRegistry 的风格，集中管理可用的 Skill，
支持按名查找、列举、转 schema（供 LLM 调用）。
"""
from __future__ import annotations

from agent.skills.base import BaseSkill


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """注册一个 Skill。"""
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' 已存在")
        self._skills[skill.name] = skill
        return self

    def get(self, name: str) -> BaseSkill | None:
        """按名查找 Skill。"""
        return self._skills.get(name)

    def all(self) -> list[BaseSkill]:
        """列出所有 Skill 实例。"""
        return list(self._skills.values())

    @property
    def names(self) -> list[str]:
        """列出所有 Skill 名称。"""
        return list(self._skills.keys())

    def schemas(self) -> list[dict]:
        """转成 OpenAI function schema 列表，供 LLM 调用。"""
        return [s.to_openai_schema() for s in self._skills.values()]
