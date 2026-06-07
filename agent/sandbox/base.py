"""
Sandbox 基类 - 安全执行环境
"""

from abc import ABC, abstractmethod


class BaseSandbox(ABC):
    """沙箱执行环境抽象基类"""

    @abstractmethod
    def execute(self, code: str, language: str = "python") -> dict:
        """在沙箱中执行代码，返回 stdout/stderr/exit_code"""
        ...
