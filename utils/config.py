import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    全局配置类
    """
    # 配置日志文件路径，用于持久化存储应用运行日志
    LOG_FILE = "logfile/app.log"
    # 如果日志文件所在的目录不存在，则自动创建目录，确保日志写入不会因路径缺失而报错
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    # 配置单个日志文件的最大字节数（这里是 5MB），通常用于配合轮转日志处理
    MAX_BYTES = 5*1024*1024,
    # 配置日志轮转时最多保留的备份文件数量，这里设置为保留 3 个历史日志文件
    BACKUP_COUNT = 3
    # 配置使用的大模型类型
    LLM_TYPE = "qwen"       # - "qwen"：调用阿里通义千问大模型
    # 配置 SQLite 记忆持久化数据库路径
    MEMORY_DB_PATH = "data/memory.db"
    # 配置 SQL Agent 使用的业务数据库路径
    SQL_AGENT_DB_PATH = "data/app.db"
  