"""
WeatherReportSkill - 示例 Skill

演示 Skill = 编排多个 Tool + 一次 LLM 润色：
先用 get_weather 工具拿数据，再用 LLM 生成一段贴心的出行建议。
"""
from agent.skills.base import BaseSkill
from agent.skills.context import SkillContext
from agent.skills.result import SkillResult


class WeatherReportSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "weather_report"

    @property
    def description(self) -> str:
        return "根据城市与日期，查询天气并生成包含出行建议的自然语言播报。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市，如 北京"},
                "date": {"type": "string", "description": "日期，如 明天"},
            },
            "required": ["city", "date"],
        }

    def execute(self, ctx: SkillContext, city: str, date: str) -> SkillResult:
        steps = []
        # 1) 编排 Tool：调用已有的 get_weather
        raw = ctx.call_tool("get_weather", query=f"{city}{date}")
        steps.append({"action": "get_weather", "result": raw})

        # 2) 编排 LLM：把原始天气润色成出行建议
        advice = ctx.llm_chat([
            {"role": "system", "content": "你是贴心的出行助手，根据天气给出简短建议。"},
            {"role": "user", "content": f"{city}{date}的天气：{raw}。请给出一句出行建议。"},
        ])
        steps.append({"action": "llm_polish", "result": advice})

        return SkillResult.ok(f"{raw}\n建议：{advice}", steps=steps)
