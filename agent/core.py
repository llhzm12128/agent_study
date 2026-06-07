"""
Agent 核心类 - 框架的中枢调度器
"""
from __future__ import annotations

from typing import Generator
from agent.llm.base import BaseLLM
from agent.memory.base import BaseMemory
from agent.tools.base import BaseTool


class Agent:
    """Agent 主类，所有模块在此汇聚"""

    def __init__(self, name: str = "MyAgent",stream_mode:bool=True):
        self.name = name
        self.llm: BaseLLM | None = None
        self.memory: BaseMemory | None = None
        self.tools: list[BaseTool] = []
        self.stream_mode = stream_mode

    def set_llm(self, llm: BaseLLM):
        """设置 LLM 通道"""
        self.llm = llm
        return self

    def set_memory(self, memory: BaseMemory):
        """设置记忆模块"""
        self.memory = memory
        return self

    def add_tool(self, tool: BaseTool):
        """注册工具"""
        self.tools.append(tool)
        return self

    def run(self, user_input: str, stream_mode: bool|None=None) -> str:
        """
        agent 执行入口 - 根据 stream_mode 判断同步输出或者流式输出
        stream_mode 优先级高于 self.stream_mode
         - 如果 stream_mode 参数不为 None，则使用 stream_mode 参数值
         - 否则，使用 self.stream_mode 的值
         - 如果最终 stream_mode 为 True，则调用 run_stream() 进行流式输出
         - 如果最终 stream_mode 为 False，则调用 run_sync() 进行同步输出
        """
        """通过stream_mode判断同步输出或者异步输出 LLM 执行任务"""
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        use_stream = stream_mode if stream_mode is not None else self.stream_mode
        if use_stream:
            # 流式输出
            return self.run_stream(user_input)
        else:
            # 同步输出
            return self.run_sync(user_input)
       
    
    def run_sync(self, user_input: str) -> str:
        """同步调用 LLM，返回完整响应"""
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        messages = self._build_messages(user_input)
        response = self.llm.chat(messages)
        if self.memory:
            self.memory.add("user", user_input)
            self.memory.add("assistant", response)
        return response

    def run_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        流式调用 LLM，逐 chunk 返回文本。
        调用方可以边接收边打印，实现打字机效果。
        """
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        messages = self._build_messages(user_input)
        full_response = ""
        for chunk in self.llm.chat_stream(messages):
            full_response += chunk
            yield chunk
        # 流式结束后，将完整对话存入记忆
        if self.memory:
            self.memory.add("user", user_input)
            self.memory.add("assistant", full_response)

    def _build_messages(self, user_input: str) -> list[dict]:
        """构建发送给 LLM 的消息列表"""
        messages = []
        system_prompt = self._build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        if self.memory:
            messages.extend(self.memory.get_context())
        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tool_desc = ""
        if self.tools:
            tool_desc = chr(10) + chr(10) + "可用工具:" + chr(10)
            for t in self.tools:
                tool_desc += f"- {t.name}: {t.description}" + chr(10)
        return f"你是 {self.name}，一个智能助手。{tool_desc}"
