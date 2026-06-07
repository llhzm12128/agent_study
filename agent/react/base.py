"""
ReAct Loop 基类 - Thought/Action/Observation 循环
"""

from abc import ABC, abstractmethod


class BaseReActLoop(ABC):
    """ReAct 循环抽象基类"""

    @abstractmethod
    def step(self, user_input: str) -> dict:
        """执行一步 ReAct"""
        ...

    @abstractmethod
    def run(self, user_input: str, max_steps: int = 10) -> str:
        """运行完整 ReAct 循环直到得出最终答案"""
        ...
