"""
Skill 基类 - 可复用能力单元
Skill 比 Tool 更高级，可编排多个 Tool + LLM 调用
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """Skill 抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, context: dict) -> Any:
        """执行 skill，context 包含当前 agent 状态"""
        ...
