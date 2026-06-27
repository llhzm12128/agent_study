# 阶段 ⑦ Skill 支持 —— 生产级设计文档（教学版）

> 本文档**只做设计与讲解，不修改任何现有项目代码**。
> 你可以照着「手搓」。每段代码都是**参考实现**，放在文档里供你抄写/改造，而不是直接落进 `agent/` 目录。
> 落地时建议的目标文件已在每节标题中注明。

---

## 0. 先厘清概念：Tool vs Skill vs Agent

很多人把 Tool 和 Skill 混为一谈，生产框架（LangChain Tool/Chain、Semantic Kernel Skill/Plugin、CrewAI、AutoGPT command）里它们是**不同抽象层级**：

| 维度 | Tool（已实现） | Skill（本阶段） |
|------|----------------|-----------------|
| 粒度 | 单一原子操作（查天气、读文件） | 一段**可复用的业务流程/能力** |
| 内部 | 一次函数调用 | 可编排**多个 Tool + 多次 LLM 调用 + 控制流** |
| 状态 | 无状态 | 可有内部状态、可读写共享 context |
| 对 LLM | 直接作为 function 暴露 | 既可作为「高级工具」暴露，也可由 Agent 主动调度 |
| 类比 | CPU 指令 | 一个函数/子程序 |

**一句话**：Tool 是"能做什么"，Skill 是"怎么把若干 Tool 组合起来完成一类任务"。

> 你现有的 `BaseSkill`（`agent/skills/base.py`）已经定义了 `name / description / execute(context)`，方向是对的。本设计在它之上补齐**生产级要件**。

---

## 1. 生产实践中 Skill 必须具备的能力（设计目标）

参考 Semantic Kernel / LangChain / CrewAI 的成熟做法，一个生产可用的 Skill 层应满足：

1. **统一抽象**：标准化的生命周期 `validate → execute → (stream)`，可被框架统一调度。
2. **可组合**：Skill 内部能调用 Tool、LLM，甚至**调用其他 Skill**（组合/嵌套）。
3. **可注册可发现**：`SkillRegistry` 集中管理，支持按名查找、列举、热插拔。
4. **参数化 & 校验**：声明输入 schema，执行前做参数校验（防止脏输入进入业务流程）。
5. **可向 LLM 暴露**：能转成 OpenAI function schema，让 ReAct/Function-Call 像调用 Tool 一样调用 Skill（"Skill as Tool"）。
6. **可观测**：执行过程产出结构化结果（成功/失败、耗时、中间步骤），便于日志与调试。
7. **隔离与安全**：与阶段⑥沙箱解耦——Skill 不直接 `os.system`，而是通过注入的 Tool/Sandbox 执行危险操作。
8. **依赖注入**：Skill 不自己 new LLM/Tool，而是由框架在执行时**注入** `SkillContext`，便于测试与替换。

---

## 2. 整体架构

```
            ┌─────────────────────────────────────────────┐
            │                   Agent                      │
            │  llm / memory / tools / skills(registry)     │
            └───────────────┬──────────────────────────────┘
                            │ 构建 schema（tools + skills）
                            ▼
            ┌─────────────────────────────────────────────┐
            │                 ReactLoop                    │
            │  chat_with_tools → 解析 tool_calls           │
            │  分发：是 Tool? 还是 Skill?                  │
            └───────┬───────────────────────┬──────────────┘
                    │                       │
              find_tool               find_skill
                    │                       │
                    ▼                       ▼
              ┌──────────┐         ┌──────────────────────┐
              │  Tool    │         │       Skill          │
              │ execute  │         │  execute(ctx, **args)│
              └──────────┘         │   ├─ 调 Tool         │
                                   │   ├─ 调 LLM          │
                                   │   └─ 调 子 Skill     │
                                   └──────────────────────┘
                                            ▲
                                   注入 SkillContext
                                   (llm / tools / skills / memory)
```

**关键设计决策：Skill as Tool（让 Skill 复用现有 Function-Call 通路）**
- 你已有的 ReAct 通过 `to_openai_schema()` 把 Tool 暴露给 LLM。
- 我们让 Skill 也能 `to_openai_schema()`，于是 **ReAct 几乎不用改**：把 `tools schema + skills schema` 合并喂给 LLM；LLM 回传 `tool_calls` 时，ReAct 先查 Tool，查不到再查 Skill。
- 这是 LangChain / Semantic Kernel 的主流做法，既统一又最小侵入。

---

## 3. 核心抽象设计（参考实现）

> 目标文件（你手搓时）：`agent/skills/base.py`（增强）、`agent/skills/context.py`（新增）、`agent/skills/result.py`（新增）

### 3.1 SkillContext —— 依赖注入容器（新增 `agent/skills/context.py`）

Skill 不应自己持有 LLM/Tool 实例，否则难以测试、难以替换。框架在调用时注入一个 context：

```python
# agent/skills/context.py  （参考实现）
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class SkillContext:
    """执行期注入给 Skill 的依赖容器（依赖注入 / 控制反转）。"""
    llm: Any = None                       # BaseLLM，Skill 内部可调 LLM
    tools: dict[str, Any] = field(default_factory=dict)   # name -> BaseTool
    skills: dict[str, Any] = field(default_factory=dict)  # name -> BaseSkill（支持嵌套）
    memory: Any = None                    # BaseMemory，可选
    shared: dict[str, Any] = field(default_factory=dict)  # 跨步骤共享的临时状态
    sandbox: Any = None                   # 阶段⑥沙箱，可选

    # —— 便捷方法：让 Skill 写起来像"调函数" ——
    def call_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' 未注入到 SkillContext")
        return self.tools[name].execute(**kwargs)

    def call_skill(self, name: str, **kwargs) -> "SkillResult":
        if name not in self.skills:
            raise KeyError(f"Skill '{name}' 未注入到 SkillContext")
        # 嵌套调用时复用同一个 context，实现组合
        return self.skills[name].run(self, **kwargs)

    def llm_chat(self, messages: list[dict]) -> str:
        if self.llm is None:
            raise RuntimeError("SkillContext 未注入 LLM")
        return self.llm.chat(messages)
```

> 设计权衡：用 `dataclass` 而非一堆构造参数，便于扩展字段、便于测试时 mock。

### 3.2 SkillResult —— 结构化结果（新增 `agent/skills/result.py`）

生产环境里 Skill 返回值绝不能只是裸字符串，要可观测、可判定成功失败：

```python
# agent/skills/result.py  （参考实现）
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SkillResult:
    success: bool
    output: Any = None                 # 最终产物（给 LLM 看的内容）
    error: str | None = None
    steps: list[dict] = field(default_factory=list)  # 中间步骤，便于调试/审计
    elapsed_ms: float | None = None

    @classmethod
    def ok(cls, output, **kw):
        return cls(success=True, output=output, **kw)

    @classmethod
    def fail(cls, error, **kw):
        return cls(success=False, error=error, **kw)

    def to_observation(self) -> str:
        """转成 ReAct 的 observation 字符串（喂回给 LLM）。"""
        if self.success:
            return str(self.output)
        return f"[Skill 执行失败] {self.error}"
```

### 3.3 BaseSkill —— 增强抽象（增强 `agent/skills/base.py`）

在你现有 `name/description/execute` 基础上，补齐 **参数 schema、校验、统一生命周期 `run()`、转 OpenAI schema**：

```python
# agent/skills/base.py  （增强版参考实现）
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Any
from agent.skills.context import SkillContext
from agent.skills.result import SkillResult

class BaseSkill(ABC):
    """Skill 抽象基类：可复用、可组合的高级能力单元。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict:
        """输入参数的 JSON Schema（默认无参）。子类可覆盖。"""
        return {"type": "object", "properties": {}, "required": []}

    # —— 业务逻辑：子类只需实现这个 ——
    @abstractmethod
    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        ...

    # —— 可选：参数校验钩子（默认基于 required 字段做最简校验）——
    def validate(self, **kwargs) -> None:
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Skill '{self.name}' 缺少必填参数: {missing}")

    # —— 框架统一入口：包裹校验 + 计时 + 异常兜底 ——
    def run(self, ctx: SkillContext, **kwargs) -> SkillResult:
        start = time.perf_counter()
        try:
            self.validate(**kwargs)
            result = self.execute(ctx, **kwargs)
        except Exception as e:
            result = SkillResult.fail(f"{type(e).__name__}: {e}")
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result

    # —— Skill as Tool：让 LLM 能像调工具一样调用 Skill ——
    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

> 设计要点：
> - 子类只实现 `execute(ctx, **kwargs)`，框架提供的 `run()` 统一处理**校验 / 计时 / 异常兜底**——这是生产代码必须的"护栏"，避免某个 Skill 抛异常把整个 ReAct 循环搞崩。
> - `to_openai_schema()` 与 `BaseTool` 同形，天然能合并进现有 Function-Call 通路。

---

## 4. SkillRegistry —— 注册与发现（新增 `agent/skills/registry.py`）

仿照你已有的 `LLMRegistry`，保持代码风格一致：

```python
# agent/skills/registry.py  （参考实现）
from __future__ import annotations
from agent.skills.base import BaseSkill

class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' 已存在")
        self._skills[skill.name] = skill
        return self

    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)

    def all(self) -> list[BaseSkill]:
        return list(self._skills.values())

    @property
    def names(self) -> list[str]:
        return list(self._skills.keys())

    def schemas(self) -> list[dict]:
        return [s.to_openai_schema() for s in self._skills.values()]
```

---

## 5. 一个真实可用的示例 Skill（新增 `agent/skills/weather_report_skill.py`）

演示 **Skill = 编排多个 Tool + 一次 LLM 润色**：先用 `get_weather` 工具拿数据，再用 LLM 生成一段贴心的出行建议。

```python
# agent/skills/weather_report_skill.py  （参考实现）
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
```

> 这就是 Skill 的价值：把"查天气 + LLM 生成建议"这套**固定流程**封装成一个可复用单元，LLM 只需调用 `weather_report(city, date)` 一次，而不是自己一步步 ReAct。

---

## 6. 与 Agent / ReActLoop 的集成（最小侵入改造点）

> 这部分是"将来落地时需要在 `core.py` / `react_loop.py` 改动的地方"，现在只列设计，不动代码。

### 6.1 `Agent`（`core.py`）需新增

```python
# 伪代码 / 改造点
self._skill_registry = SkillRegistry()

def add_skill(self, skill: BaseSkill):
    self._skill_registry.register(skill)
    return self

def _build_skill_context(self) -> SkillContext:
    return SkillContext(
        llm=self.llm,
        tools={t.name: t for t in self.tools},
        skills={s.name: s for s in self._skill_registry.all()},
        memory=self.memory,
    )
```
并在创建 `ReactLoop` 时把 skills 与 skill_context 一起传入。

### 6.2 `ReactLoop` 需改造

1. **合并 schema**：`tools_schema = [t.to_openai_schema() for t in tools] + skill_registry.schemas()`。
2. **分发执行**：解析 `tool_calls` 时——
   ```python
   tool = self.find_tool(name)
   if tool:
       observation = self._execute_tool_action(name, args)
   else:
       skill = self.skill_registry.get(name)
       if skill:
           observation = skill.run(self.skill_context, **args).to_observation()
       else:
           observation = f"未找到工具或技能: {name}"
   ```

> 由于 Skill 与 Tool 的 schema 同形、调用约定一致，ReAct 主循环改动**极小**——这正是"Skill as Tool"设计的红利。

---

## 7. 设计权衡（面试/复盘常考）

| 抉择 | 方案 A | 方案 B | 本设计选择 & 理由 |
|------|--------|--------|-------------------|
| Skill 暴露方式 | 作为独立调度层（Agent 主动选 Skill） | Skill as Tool（混入 function-call） | **B**：最小侵入、复用现有通路；A 可作为进阶 |
| 依赖获取 | Skill 自己 new LLM/Tool | 依赖注入 SkillContext | **注入**：可测试、可替换、可隔离 |
| 返回值 | 裸 str | 结构化 SkillResult | **SkillResult**：可观测、能区分成败 |
| 异常处理 | 让异常冒泡 | 框架 run() 兜底 | **兜底**：单个 Skill 失败不拖垮整个 ReAct |
| 组合方式 | 硬编码调用 | ctx.call_skill 嵌套 | **嵌套**：支持 Skill 复用 Skill |

---

## 8. 落地清单（你手搓时的 TODO）

- [ ] 新增 `agent/skills/context.py`（SkillContext）
- [ ] 新增 `agent/skills/result.py`（SkillResult）
- [ ] 增强 `agent/skills/base.py`（parameters / validate / run / to_openai_schema）
- [ ] 新增 `agent/skills/registry.py`（SkillRegistry）
- [ ] 新增 `agent/skills/weather_report_skill.py`（示例）
- [ ] 改造 `agent/core.py`：`add_skill` / `_build_skill_context`
- [ ] 改造 `agent/react/react_loop.py`：合并 schema + Tool/Skill 分发
- [ ] 在 `main.py` 注册示例 Skill 并验证：`agent.add_skill(WeatherReportSkill())`

---

## 9. 顺带提醒（与本阶段相关的既有小坑）

- `react_loop.py` 目前直接 `tool.execute()`；阶段⑥沙箱落地后，危险 Tool/Skill 应走 `ctx.sandbox`。SkillContext 已预留 `sandbox` 字段，方便将来接入。
- `ChatMemory(history=[])` 可变默认参数共享问题，建议改 `history: list | None = None`。

---

> 有疑问可以直接问我：比如「为什么不做成独立调度层」「SkillContext 和 LangChain 的 RunnableConfig 有何异同」「如何给 Skill 加权限控制」等，我可以逐一展开。
