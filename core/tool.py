"""
Agent 工具集模块

该模块负责整合并提供多种 Agent 可用的工具，包括 SQL 数据库查询。
它将这些功能封装成 LangChain 标准的工具格式，并通过 `get_tools()` 函数统一返回，以便在 Agent 中进行注册和调用。

主要功能:
- `get_tools()`: 构建并返回一个包含所有可用工具的列表。
- `sql_agent`: 用于理解和执行自然语言形式的数据库查询。
"""

from langchain.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from utils.llms import get_llm
from utils.config import Config
from utils.sql_agent import query_sql_agent
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

def get_tools():
    """
    获取 Agent 可用的工具列表。

    Returns:
        一个包含所有可用工具的列表。
    """

    @tool("sql_agent", description=
    """
    用于查询数据库中的业务数据。
    适用场景：
    - 询问用户数量、列表、详细信息（如：'有多少用户？', '显示所有北京的用户'）。
    - 询问产品信息、价格、库存（如：'最贵的产品是什么？', '笔记本电脑还有货吗？'）。
    - 任何需要从数据库中检索具体记录或统计数字的问题。
    
    注意：直接传入用户的自然语言问题，不需要生成 SQL。
    """
    )    
    def sql_agent(query: str) -> str:
        """
        根据用户自然语言问题查询数据库，返回查询结果。
        【重要】该工具已预配置了数据库连接。
        """
        try:
            logger.info(f"工具 sql_agent 被调用，查询内容: {query}")
            
            # 1. 调用原有的查询函数
            raw_result = query_sql_agent(query)
            
            # 2. 【关键修改】清洗返回值
            # query_sql_agent 返回的可能是一个字典 {'output': '...', ...} 或者包含其他元数据的对象
            # 主 Agent 需要一个纯净的字符串
            if isinstance(raw_result, dict):
                # 优先提取 'output' 字段，这是 LangChain Agent 的标准输出字段
                final_output = raw_result.get("output", str(raw_result))
            elif hasattr(raw_result, '__dict__'):
                # 如果是对象，尝试获取 output 属性
                final_output = getattr(raw_result, 'output', str(raw_result))
            else:
                final_output = str(raw_result)
            
            # 3. 确保返回的是字符串，且不要太长（防止上下文爆炸）
            if len(final_output) > 2000:
                final_output = final_output[:2000] + "\n...(结果过长，已截断)"
                
            logger.info(f"工具 sql_agent 执行成功，返回长度: {len(final_output)}")
            return final_output
            
        except Exception as e:
            error_msg = f"SQL 工具执行异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    # 将定义好的工具函数封装到列表中，作为 Agent 可调用的工具集合
    tools = [sql_agent]
    # 记录当前获取到的工具列表，方便在调试或运行中查看已注册的工具
    logger.debug(f"获取并提供的工具列表: {tools} ")
    # 返回完整的工具列表，供上层创建 Agent 时注入
    return tools