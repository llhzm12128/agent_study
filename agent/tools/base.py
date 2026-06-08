"""
Tool 基类 - 定义工具的统一接口
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """工具抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（会注入到 LLM 的 prompt 中）"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具参数的 JSON Schema 描述"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具，返回结果"""
        ...
    
    def to_openai_schema(self) -> dict:
        """转换为 OpenAI Function Schema 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
