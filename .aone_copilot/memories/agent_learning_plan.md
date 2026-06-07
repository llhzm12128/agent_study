---
name: Agent框架学习规划
description: 用户通过手搓方式学习AI Agent框架的十步规划，标记了核心模块，需跨任务长期记忆
type: insight
createdAt: 2026-06-07T14:46:00
---
用户希望通过手动实现一个完整AI Agent框架来学习agent技术栈。十步规划如下：

1. LLM通道，实现流式输出
2. 上下文管理（记忆模块）【重要】
3. 多模型切换
4. Tool Calling感知（LLM如何建议调用工具）
5. ReAct Loop【重要】
6. 沙箱执行（工具在安全容器中执行）
7. Skill支持【重要】
8. MCP stdio【重要】
9. MCP https（远程接入MCP server）
10. 长期记忆存储（向量数据库）【重要】

**Why:** 用户想由浅入深系统学习Agent开发，核心模块需要更详细说明。
**How to apply:** 后续跨任务实现时，按此顺序逐步推进，标记【重要】的模块需要更详细的设计和讲解。
EOF; __aone_exit=$?; pwd -P > '/var/folders/jt/tz3lyf0j3876yj9qljgml8080000gp/T/aone-copilot-cwd-1780814787212-j8h2w5hum7g.txt' 2>/dev/null; exit $__aone_exit