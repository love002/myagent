"""
SQL Agent 模块

本模块提供自然语言转 SQL 查询功能。
与传统的 LangChain SQL Agent 不同，这里采用"LLM 生成 SQL → 直接执行"的简化方案，
避免了嵌套 Agent 的多轮 LLM 调用，大幅降低延迟，适合语音助手场景。

主要功能:
- `query_sql_agent`: 接收自然语言问题，返回数据库查询结果。

依赖项:
- 需要通过 `config.py` 配置数据库路径 (SQL_AGENT_DB_PATH) 和 LLM 类型 (LLM_TYPE)。
- 依赖 `llms.py` 提供的 `get_chat_llm` 函数来获取大语言模型实例。
"""

import os
import re
import sqlite3
from .llms import get_chat_llm
from .config import Config
from .logger import LoggerManager

logger = LoggerManager.get_logger()

# 数据库表结构描述（注入给 LLM 让其生成准确的 SQL）
TABLE_SCHEMA = """
数据库中有两张表：

1. users（用户表）
   - id: INTEGER 主键
   - name: TEXT 姓名
   - email: TEXT 邮箱
   - age: INTEGER 年龄
   - city: TEXT 城市

2. products（产品表）
   - id: INTEGER 主键
   - name: TEXT 产品名称
   - category: TEXT 分类
   - price: REAL 价格
   - stock: INTEGER 库存
"""

SYSTEM_PROMPT_FOR_SQL = (
    "你是一个 SQL 专家。根据用户的自然语言问题，生成对应的 SQLite SQL 查询语句。\n"
    "要求：\n"
    "1. 只输出 SQL 语句本身，不要任何解释、不要 markdown 代码块标记。\n"
    "2. SQL 语句必须是 SQLite 兼容的语法。\n"
    "3. 如果用户问题与数据库无关，输出 'NO_SQL_NEEDED'。\n"
    f"{TABLE_SCHEMA}"
)


def extract_sql_from_response(response: str) -> str:
    """从 LLM 回复中提取 SQL 语句（去除 ```sql 等包裹）。"""
    # 尝试匹配 ```sql ... ```
    match = re.search(r'```(?:sql)?\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 如果没有代码块包裹，直接返回（已去除首尾空白）
    return response.strip()


def query_sql_agent(question: str) -> str:
    """
    使用 LLM 生成 SQL 并直接查询数据库。

    相比 LangChain 的 create_sql_agent（内部多次 LLM 调用），
    此方案只调用一次 LLM 生成 SQL，然后直接执行，速度提升 3-5 倍。

    Args:
        question: 自然语言查询，例如"有多少个用户？"或"显示所有北京的用户"

    Returns:
        格式化的查询结果字符串
    """
    try:
        # 1. LLM 生成 SQL
        llm = get_chat_llm(Config.LLM_TYPE, temperature=0.1)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_SQL},
            {"role": "user", "content": question},
        ]
        logger.info(f"生成 SQL: {question}")
        response = llm.invoke(messages)
        sql = extract_sql_from_response(response.content if hasattr(response, 'content') else str(response))

        if sql.upper() == 'NO_SQL_NEEDED':
            return "问题与数据库无关。"

        logger.info(f"执行 SQL: {sql}")

        # 2. 执行 SQL
        db_path = Config.SQL_AGENT_DB_PATH
        if not os.path.exists(db_path):
            return "数据库不存在，请先初始化。"

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.commit()
        finally:
            conn.close()

        # 3. 格式化结果
        if not rows:
            return "查询完成，未找到匹配的数据。"

        # 获取列名
        col_names = [desc[0] for desc in cursor.description]
        # 格式化为文本
        result_lines = []
        for row in rows:
            row_dict = dict(row)
            parts = [f"{col}: {row_dict[col]}" for col in col_names]
            result_lines.append("，".join(parts))

        formatted = "\n".join(result_lines)
        logger.info(f"SQL 查询完成，返回 {len(rows)} 条结果")
        return formatted

    except Exception as e:
        error_msg = f"查询时发生错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


def create_example_database(db_path: str):
    """
    创建一个示例 SQLite 数据库，包含一些示例表和数据。
    """
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


def ensure_database():
    """确保数据库存在，不存在则创建示例数据库。"""
    db_path = Config.SQL_AGENT_DB_PATH
    dir_path = os.path.dirname(db_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建 SQL 数据库目录: {dir_path}")
    if not os.path.exists(db_path):
        create_example_database(db_path)


# 启动时确保数据库存在
ensure_database()


if __name__ == "__main__":
    # 简单测试
    print("测试 SQL 查询...")
    answer = query_sql_agent("显示所有北京的用户")
    print(f"结果:\n{answer}")