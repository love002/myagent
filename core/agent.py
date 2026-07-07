"""
LangGraph 智能体核心模块

该模块负责整合各个组件，创建并运行一个基于 LangChain 和 LangGraph 的智能对话代理（Agent）。
它封装了模型、内存、工具和系统提示词等关键元素。
此模块是整个对话系统的大脑，协调所有后端服务以实现连贯、上下文感知的多轮对话。

统一的接口:
- `generate_response` ,用于处理用户输入并生成智能回复。
"""

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from utils.memory_sqlite import get_sqlite_saver
from utils.llms import get_chat_llm
from .prompt import get_system_prompt
from .tool import get_tools

# 获取聊天模型
chat_bot = get_chat_llm()
# 获取 SQLite Saver 唯一实例
checkpointer = get_sqlite_saver()
# 获取工具列表
tools = get_tools()
# 创建智能体
agent = create_agent(
    model=chat_bot,
    system_prompt=SystemMessage(content=get_system_prompt()),
    tools=tools,
    # context_schema=Context,
    # response_format=ToolStrategy(ResponseFormat), 
    checkpointer=checkpointer,
)

def generate_response(user_input: str, config: dict) -> str:
    """
    接收用户输入和对话历史，返回Agent的回复。
    此处的 conversation_history 应为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] 的形式。
    """    
    agent_messages_for_this_call = [{"role": "user", "content": user_input}]

    # 调用Agent，传入 messages 和 config
    result = agent.invoke({"messages": agent_messages_for_this_call}, config=config)

    # 从Agent的返回结果中提取AI的回复内容
    ai_message_content = ""
    if result and 'messages' in result and len(result['messages']) > 0:
        last_message = result['messages'][-1]
        # 确保是AI回复的消息
        if hasattr(last_message, 'content') and hasattr(last_message, 'type') and last_message.type == 'ai':
             ai_message_content = last_message.content
        elif isinstance(last_message, dict): # 如果返回的是字典格式
             ai_message_content = last_message.get('content', '')

    return ai_message_content
