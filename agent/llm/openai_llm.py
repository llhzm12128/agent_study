"""
OpenAI 兼容 LLM 实现
支持所有 OpenAI API 兼容的模型服务（通义千问、DeepSeek、GPT 等）
"""

from typing import Generator
from openai import OpenAI
from agent.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """基于 OpenAI SDK 的 LLM 实现，支持同步和流式输出"""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        """同步调用，等待完整响应后一次性返回"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """
        流式调用，逐 chunk 返回文本片段。

        流式输出的核心原理：
        - HTTP 使用 SSE (Server-Sent Events) 协议
        - 服务端生成一个 token 就推送一个 chunk
        - 客户端逐个接收并 yield 出去，实现打字机效果
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,  # 关键参数：开启流式
        )
        for chunk in stream:
            # 每个 chunk 包含一个 delta，delta.content 是本次新增的文本片段
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                yield delta_content
