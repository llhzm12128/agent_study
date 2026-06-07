"""
Memory 基类 - 定义记忆模块的统一接口
"""

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """记忆模块抽象基类"""

    @abstractmethod
    def add(self, role: str, content: str):
        """添加一条对话记录"""
        ...

    @abstractmethod
    def get_context(self) -> list[dict]:
        """获取当前上下文（用于发给 LLM）"""
        ...

    @abstractmethod
    def clear(self):
        """清空记忆"""
        ...
