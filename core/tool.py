"""
Agent 工具集模块

该模块负责整合并提供多种 Agent 可用的工具，包括 SQL 数据库查询和联网搜索。
它将功能封装成 LangChain 标准的工具格式，并通过 `get_tools()` 函数统一返回，以便在 Agent 中进行注册和调用。

主要功能:
- `get_tools()`: 构建并返回一个包含所有可用工具的列表。
- `sql_agent`: 用于理解和执行自然语言形式的数据库查询。
- `tavily_search`: 用于执行实时网络搜索，获取最新资讯。
"""

from langchain.tools import tool
from utils.sql_agent import query_sql_agent
from utils.search.tavily_search import run_tavily_search
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

            # 调用 SQL 查询（现在返回纯文本，不需额外解析）
            result = query_sql_agent(query)

            # 确保返回的是字符串，且不要太长（防止上下文爆炸）
            if len(result) > 2000:
                result = result[:2000] + "\n...(结果过长，已截断)"

            logger.info(f"工具 sql_agent 执行成功，返回长度: {len(result)}")
            return result

        except Exception as e:
            error_msg = f"SQL 工具执行异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @tool("tavily_search", description=
    """
    使用 Tavily 进行实时网络检索，回答用户关于最新资讯、新闻、事件等问题。
    适用场景：
    - 询问今天的新闻、热点事件（如：'今天有什么新闻？'）。
    - 询问实时数据、最新信息（如：'今天的天气怎么样？', '比特币现在价格多少？'）。
    - 询问你不知道的时事、名人动态等。
    - 任何需要从互联网获取最新信息的问题。

    注意：直接传入用户的问题即可，不需要添加额外修饰词。
    """
    )
    def tavily_search(query: str) -> str:
        """
        根据用户查询使用 Tavily 进行网络检索，返回最新的相关信息。

        Args:
            query: 用户关于最新资讯的问题

        Returns:
            格式化后的网络搜索结果
        """
        try:
            result = run_tavily_search(query)
            return result
        except Exception as e:
            return f"搜索失败，请检查网络连接或稍后再试。错误信息：{str(e)}"

    # 将定义好的工具函数封装到列表中，作为 Agent 可调用的工具集合
    tools = [sql_agent, tavily_search]
    # 记录当前获取到的工具列表，方便在调试或运行中查看已注册的工具
    logger.debug(f"获取并提供的工具列表: {tools} ")
    # 返回完整的工具列表，供上层创建 Agent 时注入
    return tools