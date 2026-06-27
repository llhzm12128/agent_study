"""
Skill 基类 - 可复用、可组合的高级能力单元

Skill 比 Tool 更高级，可编排多个 Tool + 多次 LLM 调用 + 控制流，
甚至调用其他 Skill（嵌套组合）。

设计要点：
- 子类只需实现 execute(ctx, **kwargs)，框架提供的 run() 统一处理
  参数校验 / 计时 / 异常兜底——避免某个 Skill 抛异常拖垮整个 ReAct 循环。
- to_openai_schema() 与 BaseTool 同形，天然能合并进现有 Function-Call 通路
  （"Skill as Tool"）。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from agent.skills.context import SkillContext
from agent.skills.result import SkillResult


class BaseSkill(ABC):
    """Skill 抽象基类：可复用、可组合的高级能力单元。"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    def parameters(self) -> dict:
        """输入参数的 JSON Schema（默认无参）。子类可覆盖。"""
        return {"type": "object", "properties": {}, "required": []}

    # —— 业务逻辑：子类只需实现这个 ——
    @abstractmethod
    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        """执行 skill 的核心业务逻辑。"""
        ...

    # —— 可选：参数校验钩子（默认基于 required 字段做最简校验）——
    def validate(self, **kwargs) -> None:
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Skill '{self.name}' 缺少必填参数: {missing}")

    # —— 框架统一入口：包裹校验 + 计时 + 异常兜底 ——
    def run(self, ctx: SkillContext, **kwargs) -> SkillResult:
        start = time.perf_counter()
        try:
            self.validate(**kwargs)
            result = self.execute(ctx, **kwargs)
        except Exception as e:
            result = SkillResult.fail(f"{type(e).__name__}: {e}")
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result

    # —— Skill as Tool：让 LLM 能像调工具一样调用 Skill ——
    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
