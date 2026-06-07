"""
框架配置
"""

import os


class Config:
    # API Key 从环境变量读取
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")

    # 以下为固定配置，直接修改此处即可
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen-plus"
    STREAM_MODE = True
    MEMORY_MAX_TOKENS = 4000
    VECTOR_DB_PATH = "./data/vector_store"
