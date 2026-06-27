"""
SkillContext - 执行期注入给 Skill 的依赖容器（依赖注入 / 控制反转）

Skill 不应自己持有 LLM/Tool 实例，否则难以测试、难以替换。
框架在调用 Skill 时注入一个 SkillContext，Skill 通过它访问 LLM/Tool/其他 Skill。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.skills.result import SkillResult


@dataclass
class SkillContext:
    """执行期注入给 Skill 的依赖容器。"""

    llm: Any = None  # BaseLLM，Skill 内部可调 LLM
    tools: dict[str, Any] = field(default_factory=dict)   # name -> BaseTool
    skills: dict[str, Any] = field(default_factory=dict)  # name -> BaseSkill（支持嵌套）
    memory: Any = None  # BaseMemory，可选
    shared: dict[str, Any] = field(default_factory=dict)  # 跨步骤共享的临时状态
    sandbox: Any = None  # 阶段⑥沙箱，可选

    # —— 便捷方法：让 Skill 写起来像"调函数" ——
    def call_tool(self, name: str, **kwargs) -> Any:
        """调用一个已注入的 Tool。"""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' 未注入到 SkillContext")
        return self.tools[name].execute(**kwargs)

    def call_skill(self, name: str, **kwargs) -> "SkillResult":
        """调用另一个 Skill（嵌套组合），复用同一个 context。"""
        if name not in self.skills:
            raise KeyError(f"Skill '{name}' 未注入到 SkillContext")
        return self.skills[name].run(self, **kwargs)

    def llm_chat(self, messages: list[dict]) -> str:
        """通过注入的 LLM 进行一次同步对话。"""
        if self.llm is None:
            raise RuntimeError("SkillContext 未注入 LLM")
        return self.llm.chat(messages)
