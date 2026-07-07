"""
SQL Agent 模块

本模块提供了一个基于 LangChain 的 SQL Agent，它能够理解自然语言查询，
并将其转换为 SQL 命令来操作数据库。模块实现了单例模式以管理 Agent 实例，
并包含创建示例数据库的功能，方便快速启动和测试。

主要功能:
- `query_sql_agent`: 接收自然语言问题，返回数据库查询结果。
- `get_sql_agent`: 提供全局唯一的 SQL Agent 实例。
- `create_example_database`: 在指定路径创建一个包含示例表 (users, products) 的 SQLite 数据库。

依赖项:
- 需要通过 `config.py` 配置数据库路径 (SQL_AGENT_DB_PATH) 和 LLM 类型 (LLM_TYPE)。
- 依赖 `llms.py` 提供的 `get_llm` 函数来获取大语言模型实例。
- 依赖 `logger.py` 提供的日志记录功能。

暴露方法：
- `query_sql_agent(question: str)`: 接收自然语言问题，返回数据库查询结果。
"""

import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
import sqlite3
from .llms import get_llm
from .config import Config
from .logger import LoggerManager

logger = LoggerManager.get_logger()
# 全局 Agent 实例
_sql_agent_instance = None

def create_example_database(db_path: str):
    """
    创建一个示例 SQLite 数据库，包含一些示例表和数据。
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建 users 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            city TEXT
        )
    ''')
    
    # 插入示例数据
    cursor.executemany(
        "INSERT OR IGNORE INTO users (name, email, age, city) VALUES (?, ?, ?, ?)",
        [
            ("张三", "zhangsan@example.com", 28, "北京"),
            ("李四", "lisi@example.com", 35, "上海"),
            ("王五", "wangwu@example.com", 22, "广州"),
            ("赵六", "zhaoliu@example.com", 40, "深圳"),
        ]
    )
    
    # 创建 products 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    ''')
    
    cursor.executemany(
        "INSERT OR IGNORE INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)",
        [
            ("笔记本电脑", "电子产品", 5999.99, 50),
            ("智能手机", "电子产品", 2999.50, 100),
            ("办公椅", "家具", 899.00, 30),
            ("咖啡机", "家电", 1500.00, 20),
        ]
    )
    
    conn.commit()
    conn.close()
    logger.info(f"示例数据库已创建，包含 users 和 products 表")

def get_sql_agent():
    """
    获取 SQL Agent 单例实例。
    该 Agent 连接到配置的 SQL_AGENT_DB_PATH 数据库。
    """
    global _sql_agent_instance
    if _sql_agent_instance is None:
        db_path = Config.SQL_AGENT_DB_PATH
        # 确保数据库目录存在
        dir_path = os.path.dirname(db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"创建 SQL 数据库目录: {dir_path}")
        
        # 如果数据库文件不存在，创建一个示例数据库
        if not os.path.exists(db_path):
            create_example_database(db_path)
            logger.info(f"创建示例数据库: {db_path}")
        # 连接数据库
        db_uri = f"sqlite:///{db_path}"
        logger.info(f"连接 SQL 数据库: {db_uri}")
        db = SQLDatabase.from_uri(db_uri)
        
        # 获取 LLM（使用默认类型）
        llm_chat, _ = get_llm(Config.LLM_TYPE)
        
        # 创建 SQL 工具包
        toolkit = SQLDatabaseToolkit(db=db, llm=llm_chat)
        
        # 创建 SQL Agent
        agent = create_sql_agent(
            llm=llm_chat,
            toolkit=toolkit,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
        )
        _sql_agent_instance = agent
        logger.info("SQL Agent 初始化完成")
    return _sql_agent_instance

def query_sql_agent(question: str) -> str:
    """
    使用 SQL Agent 执行自然语言查询，返回结果字符串。
    
    Args:
        question: 自然语言查询，例如“有多少个用户？”或“显示所有产品”
    
    Returns:
        Agent 执行后的回答字符串。
    """
    try:
        agent = get_sql_agent()
        logger.info(f"执行 SQL Agent 查询: {question}")
        result = agent.invoke({"input": question})
        output = result.get("output", str(result))
        logger.info(f"SQL Agent 查询完成")
        return output
    except Exception as e:
        error_msg = f"SQL Agent 查询时发生错误: {str(e)}"
        logger.error(error_msg)
        return error_msg

if __name__ == "__main__":
    # 简单测试
    print("测试 SQL Agent...")
    answer = query_sql_agent("张三今年几岁？")
    print(f"回答: {answer}")