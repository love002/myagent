"""
搜索模块包

提供联网搜索功能，当前支持：
- `tavily_search` - 基于 Tavily API 的实时网络搜索
"""
from .tavily_search import run_tavily_search, get_tavily_client

__all__ = ["run_tavily_search", "get_tavily_client"]
