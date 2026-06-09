
from __future__ import annotations

from typing import Generator
from agent.llm.base import BaseLLM
from agent.memory.base import BaseMemory
from agent.tools.base import BaseTool
from agent.llm.registry import LLMRegistry
import json



class ReactLoop:
    def __init__(self, llm,tools,memory=None,max_steps=10,verbose=True):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.steps:list[dict] = [] #记录每一步的思考、行动和观察
        self.verbose = True #是否打印每一步的日志

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
       
    
    def run(self, user_input: str) -> str:
        """执行完整的 ReAct 循环，返回最终回答"""
        if not self.llm:
            raise RuntimeError("未设置 LLM，请先调用 set_llm()")
        messages = self._build_messages(user_input)
        tools_schema = self._get_tools_schema()
        for step_num in range(self.max_steps):  # ReAct 循环的最大步数，防止死循环
            #1.调用LLM生成思考、行动指令和工具调用指令
            response = self.llm.chat_with_tools(messages, tools_schema)
            thought = response["content"] or ""
         
            # 2.输出thought
            if thought and self.verbose:
                print(f"Thought {step_num+1}: {thought}")
            # 3.检查是否有工具调用指令，没有则直接返回最终回答
            if not response.get("tool_calls"):
                self.steps.append({"step": step_num + 1, "type": "final_answer", "thought": thought})
                if self.verbose:
                    print(f"Final Answer: {thought}")
                self.save_to_memory(user_input, thought)
                final_answer = response["content"] or   ""      
                break
            #4.有tool_calls，需要执行工具
            messages.append({"role": "assistant", 
                             "content": thought, 
                             "tool_calls": response["tool_calls"]})
                
            # 执行工具调用指令，获取工具执行结果
            for tc in response["tool_calls"]:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])
                if self.verbose:
                    print(f"Action {step_num+1}: {tool_name} with args {tool_args}")
                #查找并执行工具
                observation = self._execute_tool_action(tool_name, tool_args)
                if(self.verbose):
                    print(f"Observation {step_num+1}: {observation}")
                #记录步骤
                self.steps.append({"step": step_num + 1, "type": "action", 
                                   "thought": thought, 
                                   "action": tool_name, 
                                   "args": tool_args, 
                                   "observation": observation})
                #追加tool消息
                messages.append({"role": "tool", 
                                 "content": str(observation), 
                                 "tool_name": tool_name,
                                 "tool_call_id": tc["id"],
                                 })
        else:
            final_answer = "推理步数超限、可能出现死循环，已强制结束对话。"
        
        if self.memory:
            self.memory.add("user", user_input)
            self.memory.add("assistant", final_answer)
        return final_answer



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
        return (f"你是一个智能助手。请一步步思考后再行动。\n"
                f"如果需要调用工具，请使用提供的工具。\n"
                f"如果已有足够信息，直接给出最终回答。"
                f"{tool_desc}")
    
    
    
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
    
    def  _execute_tool_action(self, action_name: str, action_args: dict) -> str:
        """执行 LLM 生成的行动指令，返回行动结果"""
        tool = self.find_tool(action_name)
        if not tool:
            return f"工具 {action_name} 未找到"
        try:
            result = tool.execute(**action_args)
            return str(result)
        except Exception as e:
            return f"执行工具 {action_name} 时出错: {str(e)}"
        
    def save_to_memory(self, user_input: str, final_answer: str):
        """将对话内容保存到记忆模块"""
        if self.memory:
            self.memory.add("user", user_input)
            self.memory.add("assistant", final_answer)

    