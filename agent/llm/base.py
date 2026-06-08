"""
LLM 基类 - 定义 LLM 通道的统一接口
"""
from __future__ import annotations

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

    @abstractmethod
    def chat_with_tools(self, messages: list[dict], tools: list[BaseTool]) -> dict:
        """支持工具调用的对话接口，返回 LLM 响应和工具调用指令
        
        返回值结构：
        {
            "content": str, ## LLM 生成的文本内容，我工具调用时会放在 content 字段里
            "tool_call": list[dict] ## LLM 生成的工具调用指令列表，每条指令包含 tool_name 和 tool_args 两个字段,
        }
        """
        
        ...
