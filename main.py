import asyncio
import sys
import os
import threading
import time
import json
from typing import AsyncGenerator
from utils.asr import RealTimeASR
from utils.logger import LoggerManager
from utils.TTS.tts import stream_tts_from_generator

logger = LoggerManager.get_logger()

# --- 导入Agent 和 Memory ---
# 为了防止循环导入和不必要的重复初始化，我们不在全局作用域导入。
# 而是在首次需要时，动态导入并缓存agent实例。
_agent_module = None
_memory_module = None # 新增：导入memory模块

def get_agent_module():
    """获取Agent模块的辅助函数，实现懒加载。"""
    global _agent_module

    if _agent_module is None:
        # 在导入前，将当前脚本所在目录添加到系统路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            import core.agent as loaded_agent_module
            _agent_module = loaded_agent_module
            logger.debug("[Agent] 成功从 agent.py 加载智能体模块。")
            print("")
        except ImportError as e:
            logger.error(f"[Agent] 导入失败: {e}")
            raise
    return _agent_module

# 新增：获取Memory模块的辅助函数
def get_memory_module():
    """获取Memory模块的辅助函数，实现懒加载。"""
    global _memory_module
    if _memory_module is None:
        # 在导入前，将当前脚本所在目录添加到系统路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            import utils.memory_sqlite as loaded_memory_module # 假设您的memory模块文件名为 memory.py
            _memory_module = loaded_memory_module
            logger.debug("[Memory] 成功从 memory.py 加载记忆模块。")
        except ImportError as e:
            logger.error(f"[Memory] 导入失败: {e}")
            raise
    return _memory_module

# --- 全局变量 ---
CONVERSATION_THREAD_ID = "default_conversation_001"
asr_system = None # 将asr_system定义为全局变量，以便在其他函数中访问

# --- LLM 交互逻辑 ---

async def get_llm_response(user_input: str) -> str:
    """
    使用Agent模块来获取回复。
    """
    # --- 核心改动：获取Agent和Memory模块 ---
    agent_module = get_agent_module()
    memory_module = get_memory_module()

    # 获取检查点保存器
    checkpointer = memory_module.get_sqlite_saver()

    # 定义配置，其中包含唯一的会话ID
    config = {"configurable": {"thread_id": CONVERSATION_THREAD_ID}}

    # --- 核心改动：调用Agent，传入配置 ---
    # 注意：现在不需要传递 conversation_history 了，
    # Agent 会通过 checkpointer 和 thread_id 自动加载历史
    llm_reply = agent_module.generate_response(user_input, config=config)

    # 移除：更新 conversation_history 的逻辑
    # conversation_history.append({"role": "user", "content": user_input})
    # conversation_history.append({"role": "assistant", "content": llm_reply})
    
    return llm_reply

async def text_to_speech_streamer(text: str) -> AsyncGenerator[str, None]:
    """
    将一段文本按句子或词语分割，模拟流式输出，以便TTS模块可以流式播放。
    """
    import re
    sentences = re.split(r'([。！？.!?])', text)
    full_sentences = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
        if sentence.strip():
            full_sentences.append(sentence)
    
    for sentence in full_sentences:
        yield sentence
        await asyncio.sleep(0.01) # 模拟一点延迟，让TTS有节奏地播放


async def handle_speech_response(llm_reply: str):
    """
    调用TTS模块播放LLM的回复
    """
    global asr_system # 让函数可以访问全局的asr_system

    print(f"\n[AI回复]:\n {llm_reply}\n[正在播放...]")
    
    # --- 核心改动：在TTS开始前暂停ASR ---
    logger.debug("[TTS开始播放，暂停ASR监听...]")
    asr_system.pause_listening() # 假设您的RealTimeASR类有这个方法

    try:
        # 调用您TTS模块中的流式播放函数
        await stream_tts_from_generator(
            text_generator=text_to_speech_streamer(llm_reply)
        )
    except Exception as e:
        print(f"[TTS Error]: {e}")
    finally:
        # --- 核心改动：TTS结束后恢复ASR ---
        logger.debug("[TTS播放结束，恢复ASR监听...]")
        asr_system.resume_listening() # 假设您的RealTimeASR类有这个方法

def on_asr_complete(recognized_text: str):
    """
    ASR模块识别完成后，会调用此函数。
    这是整个流程的关键连接点。
    """
    # 过滤掉无意义的内容
    if not recognized_text or "[无有效内容]" in recognized_text or "[ASR 无输出]" in recognized_text:
        return

    # --- 核心改动：移除is_tts_playing检查，因为它不再是必需的 ---
    # 现在ASR在TTS播放时已经完全停止，不会产生新的回调。
    logger.info(f"\n[已识别]: {recognized_text}")

    # 启动一个新线程来处理LLM和TTS，以免阻塞ASR的实时监听
    def process_and_respond():
        # 由于 get_llm_response 是异步的，我们需要在一个新的事件循环中运行它
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def process():
            try:
                llm_reply = await get_llm_response(recognized_text)
                await handle_speech_response(llm_reply)
            except Exception as e:
                logger.error(f"处理流程出错: {e}")

        try:
            loop.run_until_complete(process())
        finally:
            loop.close()

    thread = threading.Thread(target=process_and_respond)
    thread.daemon = True
    thread.start()

def main():
    global asr_system # 声明使用全局变量
    logger.info("------ 👾 正在启动语音助手👾 ------\n")

    # 创建ASR实例，并传入回调函数
    asr_system = RealTimeASR(on_recognized_callback=on_asr_complete)

    try:
        # 启动ASR监听
        asr_system.start_listening()
    except KeyboardInterrupt:
        logger.info("\n\n用户中断，正在退出...")
    finally:
        asr_system.listening = False
        print("\n")
        logger.info("🌴🌴🌴     测试结束，期待与您的下次相遇!")

if __name__ == "__main__":
    main()