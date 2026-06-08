from agent.llm.base import BaseLLM

class LLMRegistry:
    """
    LLM 注册表 - 管理可用的 LLM 实现
    """
    def __init__(self):
        self._models :dict[str,BaseLLM] = {}
        self.current_model: str | None = None

    def register(self, name: str, llm_cls: BaseLLM):
        """注册一个 LLM 实现"""
        self._models[name] = llm_cls
        if self.current_model is None:
            self.current_model = name

    def switch(self, name: str):
        """切换当前使用的 LLM"""
        if name not in self._models:
            raise ValueError(f"LLM {name} 未注册")
        self.current_model = name

    @property
    def current(self) -> BaseLLM:
        """获取当前使用的 LLM 实例"""
        if self.current_model is None:
            raise RuntimeError("没有可用的 LLM 模型")
        return self._models[self.current_model]

    @property
    def current_name(self) -> str:
        """获取当前使用的 LLM 名称"""
        return self.current_model or ""
    
    @property
    def list_models(self) -> list[str]:
        """列出所有注册的 LLM 模型名称"""
        return list(self._models.keys())