from agent.memory.base import BaseMemory;

class ChatMemory(BaseMemory):
    """聊天记忆模块，记录对话历史"""

    def __init__(self, max_rounds: int = 3, history: list[dict] = []):
        super().__init__()
        self.max_rounds = max_rounds
        self.history = history

    def add(self, role: str, content: str):
        """添加一轮对话，role: user/assistant"""
        if len(self.history) >= self.max_rounds:
            self.history.pop(0)
        self.history.append({"role": role, "content": content})

    def get_context(self) -> list[dict]:
        """获取当前对话历史，注意控制总 token 数不超过 max_tokens"""
        # 这里简单实现为返回全部历史，实际应用中可以根据 token 数进行截断
        return self.history
    
    def clear(self):
        """清空对话历史"""
        self.history = []