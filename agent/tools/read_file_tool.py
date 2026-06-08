
from agent.core import Agent
from agent.tools.base import BaseTool
class ReadFileTool(BaseTool):
  

    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "读取文件内容，参数是文件路径"


    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "文件路径",
                }
            },
            "required": ["query"],
        }

    def execute(self, query: str) -> str:
        # 这里直接返回一个固定的文件内容，实际应用中可以读取文件获取真实数据
        return f"文件 {query} 的内容是：这是一个示例文件。"