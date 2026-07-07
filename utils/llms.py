"""
LLM 模型工厂
- 对外接口:
    - get_llm(llm_type, temperature)
    - get_chat_model(llm_type, temperature)
    - get_embedding_model(llm_type)
"""

import os 
from dotenv import load_dotenv
from typing import Any, Tuple
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from functools import lru_cache
from .logger import LoggerManager

load_dotenv()
logger = LoggerManager.get_logger()

# 默认 LLM 类型
DEFAULT_LLM_TYPE = "deepseek"
# 默认温度为 0
DEFAULT_TEMPERATURE = 0.7

class LLMInitializationError(Exception):
    pass

# ==========================================
# 1. 配置中心 
# ==========================================

# 存放所有模型
MODEL_CONFIGS = {
    # 兼容 OpenAI 协议
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env_name": "DASHSCOPE_API_KEY", 
        "chat_model": "qwen3.7-plus",
        "embedding_model": "text-embedding-v4" 
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env_name": "DEEPSEEK_API_KEY",
        "chat_model": "deepseek-chat",
        "embedding_model": "deepseek-chat"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_name": "OPENAI_API_KEY",
        "chat_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-3-small"
    },
    # 本地 Ollama (通常不需要 key)
    "ollama": {
        "base_url": "http://localhost:11434/v1", # Ollama 新版本的兼容接口
        "api_key_env_name": None, # 不需要 Key
        "chat_model": "llama3",
        "embedding_model": "nomic-embed-text"
    }
}

# ==========================================
# 2. 工厂类 (引入 lru_cache 和 解耦)
# ==========================================

class ModelFactory:
    @staticmethod
    def _get_creds(llm_type: str) -> dict:
        # 判断输入模型类型是否支持
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的类型: {llm_type}")
        
        cfg = MODEL_CONFIGS[llm_type]
        api_key = os.getenv(cfg["api_key_env_name"])
        
        if not api_key:
            logger.warning(f"环境变量 {cfg['api_key_env_name']} 未设置，可能使用空 Key 导致失败。")
            api_key = "empty-key" # 占位，让后续 API 调用报错更明确
            
        return {
            "base_url": cfg["base_url"],
            "api_key": api_key,
            "chat_model": cfg["chat_model"],
            "embedding_model": cfg["embedding_model"]
        }

    @staticmethod
    @lru_cache(maxsize=16)
    def get_chat_model(llm_type: str = DEFAULT_LLM_TYPE, temperature: float = DEFAULT_TEMPERATURE) -> Any:
        """获取 Chat 模型 (带缓存)"""
        try:
            creds = ModelFactory._get_creds(llm_type)
            logger.info(f"初始化 Chat 模型: {llm_type} ({creds['chat_model']})")
            
            return ChatOpenAI(
                base_url=creds["base_url"],
                api_key=creds["api_key"],
                model=creds["chat_model"],
                temperature=temperature,
                timeout=30,
                max_retries=2
            )
        except Exception as e:
            # 如果指定模型失败，且不是默认模型，尝试降级到默认模型
            if llm_type != DEFAULT_LLM_TYPE:
                logger.warning(f"模型 {llm_type} 初始化失败，尝试降级到 {DEFAULT_LLM_TYPE}: {e}")
                return ModelFactory.get_chat_model(DEFAULT_LLM_TYPE, temperature)
            raise LLMInitializationError(f"Chat 模型初始化失败 ({llm_type}): {e}")

    @staticmethod
    @lru_cache(maxsize=16)
    def get_embedding_model(llm_type: str = DEFAULT_LLM_TYPE) -> Any:
        """获取 Embedding 模型 (带缓存)"""
        try:
            creds = ModelFactory._get_creds(llm_type)
            logger.info(f"初始化 Embedding 模型: {llm_type} ({creds['embedding_model']})")
            
            if llm_type == "qwen":
                logger.info("检测到 Qwen 类型，使用 DashScope 原生 Embeddings...")
                return DashScopeEmbeddings(
                    dashscope_api_key=creds["api_key"],
                    model=creds["embedding_model"],
                )
            return OpenAIEmbeddings(
                base_url=creds["base_url"],
                api_key=creds["api_key"],
                model=creds["embedding_model"],
                timeout=30
            )
        except Exception as e:
            # 同样的降级逻辑
            if llm_type != DEFAULT_LLM_TYPE:
                logger.warning(f"Embedding 模型 {llm_type} 初始化失败，尝试降级到 {DEFAULT_LLM_TYPE}: {e}")
                return ModelFactory.get_embedding_model(DEFAULT_LLM_TYPE)
            raise LLMInitializationError(f"Embedding 模型初始化失败 ({llm_type}): {e}")


# ==========================================
# 3. 对外接口 
# ==========================================

def get_llm(llm_type: str = DEFAULT_LLM_TYPE, temperature: float = DEFAULT_TEMPERATURE) -> Tuple[Any, Any]:
    """
    同时返回 Chat 和 Embedding 实例。
    但底层利用了缓存，不会重复初始化。
    """
    chat = ModelFactory.get_chat_model(llm_type, temperature)
    embed = ModelFactory.get_embedding_model(llm_type)
    return chat, embed

def get_chat_llm(llm_type: str = DEFAULT_LLM_TYPE, temperature: float = DEFAULT_TEMPERATURE) -> Any:
    """
    仅获取 Chat 实例 (推荐在新代码中使用)
    """
    return ModelFactory.get_chat_model(llm_type, temperature)

def get_embedding_llm(llm_type: str = DEFAULT_LLM_TYPE) -> Any:
    """
    仅获取 Embedding 实例 (推荐在新代码中使用)
    """
    return ModelFactory.get_embedding_model(llm_type)