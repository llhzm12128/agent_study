"""
框架配置
"""

import os


class Config:
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
    MEMORY_MAX_TOKENS = 4000
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
