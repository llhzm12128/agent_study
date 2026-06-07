"""
MCP (Model Context Protocol) 基类
支持 stdio 本地通信和 HTTPS 远程通信
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMCPClient(ABC):
    """MCP 客户端抽象基类"""

    @abstractmethod
    def connect(self):
        """建立连接"""
        ...

    @abstractmethod
    def list_tools(self) -> list[dict]:
        """获取 MCP Server 提供的工具列表"""
        ...

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用 MCP Server 上的工具"""
        ...

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        ...
