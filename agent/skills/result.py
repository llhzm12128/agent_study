"""
SkillResult - Skill 执行的结构化结果

生产环境里 Skill 返回值不应只是裸字符串，而要可观测、可判定成功失败、
可记录中间步骤，便于日志与调试。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    success: bool
    output: Any = None  # 最终产物（给 LLM 看的内容）
    error: str | None = None
    steps: list[dict] = field(default_factory=list)  # 中间步骤，便于调试/审计
    elapsed_ms: float | None = None

    @classmethod
    def ok(cls, output, **kw) -> "SkillResult":
        return cls(success=True, output=output, **kw)

    @classmethod
    def fail(cls, error, **kw) -> "SkillResult":
        return cls(success=False, error=error, **kw)

    def to_observation(self) -> str:
        """转成 ReAct 的 observation 字符串（喂回给 LLM）。"""
        if self.success:
            return str(self.output)
        return f"[Skill 执行失败] {self.error}"
