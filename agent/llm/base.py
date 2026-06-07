"""
LLM 基类 - 定义 LLM 通道的统一接口
"""

from abc import ABC, abstractmethod
from typing import Generator


class BaseLLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """同步对话，返回完整响应"""
        ...

    @abstractmethod
    def chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """流式对话，逐 token 返回"""
        ...
