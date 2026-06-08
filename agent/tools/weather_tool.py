from agent.core import Agent
from agent.tools.base import BaseTool   

class WeatherTool(BaseTool):



    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return  "查询天气，参数是一个字符串，格式为：城市+日期，例如：北京明天"


    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询字符串，格式为：城市+日期，例如：北京明天",
                }
            },
            "required": ["query"],
        }

    def execute(self, query: str) -> str:
        # 这里直接返回一个固定的天气信息，实际应用中可以调用天气API获取真实数据
        return f"{query}的天气是晴天，温度25度。"