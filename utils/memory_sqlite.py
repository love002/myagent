"""
这是一个基于 SQLite 的记忆持久化模块，负责初始化和提供全局唯一的 SqliteSaver 实例。

该模块封装了对 LangGraph 检查点数据库的访问，实现了单例模式以确保在整个应用
生命周期内共享同一个数据库连接。

主要功能包括：
- 根据配置自动创建数据库目录。
- 初始化并启用 WAL 模式的 SQLite 连接。
- 提供一个全局的 SqliteSaver 单例实例供其他模块存取检查点。
- 提供清空数据库的功能，通常用于测试或重置。

暴露函数：
- 当前仅暴露一个 get_sqlite_saver() 函数，用于获取 SQLiteSaver 单例实例。
"""
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from .config import Config
from .logger import LoggerManager

logger = LoggerManager.get_logger()
# 全局变量，存储 SQLiteSaver 的唯一实例
_sqlite_saver_instance = None

def get_sqlite_saver() -> SqliteSaver:
    """
    获取 SQLiteSaver 单例实例，连接到配置的 MEMORY_DB_PATH。
    """
    global _sqlite_saver_instance

    # 如果实例不存在，则创建
    if _sqlite_saver_instance is None:
        db_path = Config.MEMORY_DB_PATH
        ensure_db_dir(db_path)
        logger.info(f"初始化 SQLiteSaver，数据库路径: {db_path}")
        import sqlite3
        # 创建 SQLite 连接，设置 check_same_thread=False 以支持多线程使用
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # 启用 WAL 模式以提高并发性能
        conn.execute("PRAGMA journal_mode=WAL;")
        # 创建 SqliteSaver 实例
        _sqlite_saver_instance = SqliteSaver(conn)
    return _sqlite_saver_instance

def ensure_db_dir(db_path: str):
    """确保数据库文件所在目录存在"""
    dir_path = os.path.dirname(db_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建数据库目录: {dir_path}")

def clear_memory():
    """
    清空记忆数据库（用于测试或重置）。
    注意：这将删除所有检查点数据。
    """
    db_path = Config.MEMORY_DB_PATH
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"已删除记忆数据库文件: {db_path}")
        global _sqlite_saver_instance
        _sqlite_saver_instance = None
    else:
        logger.info("记忆数据库文件不存在，无需清除。")
