"""
Tavily 网络检索工具

利用 Tavily API 进行实时网络搜索，获取最新资讯。
在语音助手场景下，搜索结果会被 TTS 朗读给用户，因此输出尽量简洁清晰。

环境变量要求:
    TAVILY_API_KEY - Tavily API 密钥（已配置在 .env 中）
"""
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
from tavily import TavilyClient
from utils.logger import LoggerManager

# 加载环境变量
load_dotenv()

# 获取日志记录器
logger = LoggerManager.get_logger()

# 从环境变量获取 Tavily API 密钥
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY 环境变量未设置，Tavily 搜索可能无法正常工作。")

# 当前日期（注入到搜索查询中，避免 Tavily 返回旧结果）
_CURRENT_DATE = datetime.now().strftime("%Y年%m月%d日")

# Tavily 客户端单例
_client: Optional[TavilyClient] = None

# 匹配日期模糊关键词的正则
_DATE_AMBIGUOUS_KEYWORDS = [
    "今天", "明天", "昨天", "前天", "后天",
    "今年", "去年", "明年",
    "本月", "本月", "上月", "下月",
    "星期", "周几", "几号", "日期", "几月",
    "现在", "当前", "最近",
    "news", "today", "weather",
    "最新", "实时", "最近", "新闻",
]


def _needs_date_context(query: str) -> bool:
    """判断查询中是否包含日期模糊关键词，需要附加当前日期上下文。"""
    for kw in _DATE_AMBIGUOUS_KEYWORDS:
        if kw in query:
            return True
    return False


def _enrich_query_with_date(query: str) -> str:
    """
    如果查询中包含日期模糊关键词，自动拼接当前日期以提升搜索结果准确性。

    例如:
        "今天有什么新闻" → "今天(2026年7月7日) 有什么新闻"
        "最新科技资讯"  → "2026年7月 最新科技资讯"
    """
    if _needs_date_context(query):
        enriched = re.sub(
            r'(今天|明天|昨天|最近|最新|现在)',
            f'\\1({_CURRENT_DATE})',
            query,
            count=1,
        )
        if enriched == query:
            # 如果没命中替换逻辑，直接在开头拼日期
            enriched = f"{_CURRENT_DATE} {query}"
        logger.info(f"查询已增强日期上下文: '{query}' → '{enriched}'")
        return enriched
    return query


def get_tavily_client() -> TavilyClient:
    """
    获取 Tavily 客户端单例实例。

    Returns:
        TavilyClient 实例

    Raises:
        ValueError: 当 TAVILY_API_KEY 未设置时抛出
    """
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY 环境变量未设置，请先设置该变量以使用 Tavily 搜索。")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info("Tavily 客户端初始化成功。")
    return _client


def run_tavily_search(query: str, max_results: int = 5, search_depth: str = "advanced") -> str:
    """
    使用 Tavily 执行网络搜索，返回格式化后的搜索结果。

    针对语音播报场景做了优化：
    - 结果按序号排列，内容简洁
    - 每条结果包含标题和内容摘要
    - 过长内容自动截断
    - 自动拼接当前日期上下文，避免返回旧缓存

    Args:
        query: 搜索查询字符串
        max_results: 最大返回结果数量，默认为 5
        search_depth: 搜索深度，"advanced"(默认) 或 "basic"

    Returns:
        格式化后的搜索结果字符串，每条包含标题和摘要。
        如果发生错误，返回错误提示信息。
    """
    try:
        client = get_tavily_client()
        # 自动附加日期上下文
        enriched_query = _enrich_query_with_date(query)
        logger.info(f"执行 Tavily 搜索: {enriched_query}")

        # 使用 advanced 搜索深度以获得更准确的结果
        response = client.search(
            enriched_query,
            max_results=max_results,
            search_depth=search_depth,
        )

        # 提取搜索结果
        results = response.get("results", [])
        if not results:
            return "未找到相关结果。"

        # 格式化输出（简洁，适合 TTS 朗读）
        formatted = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "无标题")
            content = res.get("content", "").strip()
            # 限制内容长度，避免 TTS 朗读过长
            if len(content) > 200:
                content = content[:200] + "..."
            formatted.append(f"{i}. {title}: {content}")

        result_text = "\n".join(formatted)
        logger.info(f"Tavily 搜索完成，共找到 {len(results)} 条结果。")
        return result_text

    except Exception as e:
        error_msg = f"Tavily 搜索时发生错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


if __name__ == "__main__":
    # 简单测试
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "今天有什么新闻"
    print(f"测试 Tavily 搜索: {query}\n")
    print(run_tavily_search(query, max_results=3))
