"""
Agent 框架入口 - Step 1: LLM 通道 + 流式输出演示
"""

import sys
from config import Config
from agent.core import Agent
from agent.llm.openai_llm import OpenAILLM


def main():
    # 1. 创建 LLM 实例
    if not Config.LLM_API_KEY:
        print("请先设置环境变量 LLM_API_KEY")
        print("  export LLM_API_KEY=sk-xxx")
        return


    llm = OpenAILLM(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL,
    )

    # 2. 创建 Agent 并挂载 LLM
    agent = Agent(name="StudyAgent", stream_mode=Config.STREAM_MODE)
    agent.set_llm(llm)

    print(f"Agent [{agent.name}] ready | model: {Config.LLM_MODEL}| stream_mode: {Config.STREAM_MODE}")
    print("输入 quit 退出, 输入 stream: 前缀使用流式输出")
    print("-" * 50)

    # 3. 交互循环
    while True:
        try:
            user_input = input(chr(10) + "You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(chr(10) + "Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break

        result = agent.run(user_input)

        # 判断是否使用流式输出
        if agent.stream_mode:
            print("Assistant> ", end="", flush=True)
            for chunk in result:
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"Assistant> {result}")


if __name__ == "__main__":
    main()
