"""
Agent 核心类 - 框架的中枢调度器
"""
from __future__ import annotations

from typing import Generator
from agent.llm.base import BaseLLM
from agent.memory.base import BaseMemory
from agent.tools.base import BaseTool
from agent.llm.registry import LLMRegistry
import json
from agent.react.react_loop import ReactLoop
from agent.skills.base import BaseSkill
from agent.skills.registry import SkillRegistry
from agent.skills.context import SkillContext


class Agent:
    """Agent 主类，所有模块在此汇聚"""

    def __init__(self, name: str = "MyAgent",stream_mode:bool=True):
        self.name = name
        self.llm: BaseLLM | None = None
        self.memory: BaseMemory | None = None
        self.tools: list[BaseTool] = []
        self.stream_mode = stream_mode
        self._llm_registry: LLMRegistry | None = None
        self._skill_registry: SkillRegistry = SkillRegistry()

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

    def add_skill(self, skill: BaseSkill):
        """注册技能（Skill）"""
        self._skill_registry.register(skill)
        return self

    def _build_skill_context(self) -> SkillContext:
        """构建注入给 Skill 的依赖容器（依赖注入）"""
        return SkillContext(
            llm=self.llm,
            tools={t.name: t for t in self.tools},
            skills={s.name: s for s in self._skill_registry.all()},
            memory=self.memory,
        )

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
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        # 有工具或技能时走 ReAct 循环
        if self.tools or self._skill_registry.all():
            react = ReactLoop(
                llm=self.llm,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                skill_registry=self._skill_registry,
                skill_context=self._build_skill_context(),
            )
            return react.run(user_input)
        else:
            # 无工具无技能时退化为流式输出
            return "".join(self.run_stream(user_input))


    '''
    def run_sync(self, user_input: str) -> str:
        """同步调用 LLM，返回完整响应"""
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        messages = self._build_messages(user_input)
        tools_schema = self._get_tools_schema()
        for _ in range(10):  # 最多允许 LLM 调用工具 10 轮，防止死循环
            #分析用户输入和对话历史，生成工具调用指令
            response = self.llm.chat_with_tools(messages, tools_schema)
            if not response.get("tool_calls"):
                # LLM 没有生成工具调用指令，说明对话结束
                final_answer = response["content"] or ""
                break
            # LLM 生成工具调用指令，需要执行工具
            messages.append({"role": "assistant", "content": response["content"],
                             "tool_calls": response["tool_calls"]})
            # 执行工具调用指令，获取工具执行结果
            tool_results = self._execute_tool_calls(response["tool_calls"])
            messages.extend(tool_results)
        else:
            final_answer = "工具调用过多，可能出现死循环，已强制结束对话。"
        
        if self.memory:
            self.memory.add("user", user_input)
            self.memory.add("assistant", final_answer)
        return final_answer
        '''

    def run_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        流式调用 LLM，逐 chunk 返回文本。
        调用方可以边接收边打印，实现打字机效果。
        """
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        #执行之前，构建系统提示词和用户输入的消息列表
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
    
    def register_llm(self, name: str, llm_cls: BaseLLM):
        """注册一个 LLM 实现"""
        if self._llm_registry is None:
            self._llm_registry = LLMRegistry()
        self._llm_registry.register(name, llm_cls)
        self.llm = self._llm_registry.current
        return self
    
    def switch_llm(self, name: str):
        """切换当前使用的 LLM"""
        if self._llm_registry is None:
            raise RuntimeError("没有注册任何 LLM 模型")
        self._llm_registry.switch(name)
        self.llm = self._llm_registry.current
        return self
    
    def _get_tools_schema(self) -> list[dict]:
        """获取工具调用的 schema 列表，供 LLM 生成工具调用指令时参考"""
        if not self.tools:
            return None
        return [tool.to_openai_schema() for tool in self.tools]
    
    def find_tool(self, tool_name: str) -> BaseTool | None:
        """根据工具名称查找工具实例"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """执行 LLM 生成的工具调用指令，返回工具执行结果列表"""
        results = []
        for call in tool_calls:
            tool_name = call["function"]["name"]
            tool_args = json.loads(call["function"]["arguments"])
            tool = self.find_tool(tool_name)
            if not tool:
                results.append({"tool_name": tool_name, "error": "工具未找到"})
                continue
            try:
                result = tool.execute(**tool_args)
                results.append({"role": "tool", 
                                "tool_call_id": call["id"],
                                 "content": str(result),
                                 "tool_name": tool_name,
                                 })
            except Exception as e:
                results.append({"tool_name": tool_name, "error": str(e)})
        return results
    
          
    

