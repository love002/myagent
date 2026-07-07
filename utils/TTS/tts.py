"""
完整的 TTS 模块

"""

#!/usr/bin/env python3
import asyncio
import json
import logging
import uuid
import os
from typing import AsyncGenerator, Optional
import io
import time
import threading
import websockets
import pyaudio
import subprocess
import queue
from .protocols import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

# ================= 配置区域 (硬编码输入) =================
CONFIG = {
    "APP_ID": "7474732800",          #  App ID
    "ACCESS_TOKEN": "hxl7jXIJw7_BSD9FCur07oZYvqhRvHHT", # Access Token
    "VOICE_TYPE": "zh_female_sajiaonvyou_moon_bigtts", # 音色
    "OUTPUT_FORMAT": "mp3",                # 输出格式: mp3, wav, ogg
    "SAMPLE_RATE": 24000,                  # 采样率
    "ENDPOINT": "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
    "SPEED": 2, 
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLM_TTS_Stream")

# =======================================================

def get_resource_id(voice: str) -> str:
    if voice.startswith("S_"):
        return "volc.megatts.default"
    return "volc.service_type.10029"

class TTSEngine:
    def __init__(self):
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.is_connected = False
        # 【关键修复】添加锁，防止多个协程同时调用 recv
        self.recv_lock = asyncio.Lock()
        self._session_finished = asyncio.Event()

    async def connect(self):
        headers = {
            "X-Api-App-Key": CONFIG["APP_ID"],
            "X-Api-Access-Key": CONFIG["ACCESS_TOKEN"],
            "X-Api-Resource-Id": get_resource_id(CONFIG["VOICE_TYPE"]),
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

        logger.info(f"Connecting to {CONFIG['ENDPOINT']}...")
        try:
            self.websocket = await websockets.connect(
                CONFIG["ENDPOINT"], 
                additional_headers=headers, 
                max_size=10 * 1024 * 1024
            )
            log_id = self.websocket.response.headers.get('x-tt-logid', 'unknown')
            logger.info(f"Connected successfully. LogID: {log_id}")
            
            await start_connection(self.websocket)
            await wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.ConnectionStarted)
            self.is_connected = True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def start_session(self):
        if not self.is_connected:
            raise RuntimeError("Not connected to TTS server")

        self.session_id = str(uuid.uuid4())
        self._session_finished.clear() # 重置结束事件
        
        base_request = {
            "user": {"uid": str(uuid.uuid4())},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": CONFIG["VOICE_TYPE"],
                "audio_params": {
                    "format": CONFIG["OUTPUT_FORMAT"],
                    "sample_rate": CONFIG["SAMPLE_RATE"],
                    "enable_timestamp": False,
                },
                "additions": json.dumps({"disable_markdown_filter": True}),
            },
            "event": EventType.StartSession
        }

        await start_session(self.websocket, json.dumps(base_request).encode(), self.session_id)
        await wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.SessionStarted)
        logger.debug("Session started.")

    async def send_text_chunk(self, text_chunk: str):
        if not self.session_id:
            raise RuntimeError("No active session")
        
        base_request = {
            "user": {"uid": str(uuid.uuid4())},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": CONFIG["VOICE_TYPE"],
                "text": text_chunk,
                "audio_params": {
                    "format": CONFIG["OUTPUT_FORMAT"],
                    "sample_rate": CONFIG["SAMPLE_RATE"],
                },
                "additions": json.dumps({"disable_markdown_filter": True}),
            },
            "event": EventType.TaskRequest
        }
        
        await task_request(self.websocket, json.dumps(base_request).encode(), self.session_id)

    async def end_session(self):
        if self.session_id:
            logger.debug("Sending FinishSession request...")
            await finish_session(self.websocket, self.session_id)
            self.session_id = None
            # 注意：这里不等待 SessionFinished 事件，而是由 consumer 任务检测到该事件后处理
            # 这样可以避免 recv 冲突

    async def close(self):
        if self.websocket:
            # 如果 session 还没结束，尝试结束它
            if self.session_id:
                try:
                    await finish_session(self.websocket, self.session_id)
                except websockets.exceptions.ConnectionClosed:
                    pass
                except:
                    pass

            try:
                await finish_connection(self.websocket)
                # 尝试等待连接关闭事件，但不强求
                try:
                    await asyncio.wait_for(
                        wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.ConnectionFinished),
                        timeout=2.0
                    )
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass
            except websockets.exceptions.ConnectionClosed:
                logger.debug("Connection already closed during finish")
            except Exception as e:
                logger.warning(f"Error during connection finish: {e}")

            try:
                await self.websocket.close()
            except websockets.exceptions.ConnectionClosed:
                pass

            self.is_connected = False
            logger.info("Connection closed.")

    # 【关键修复】安全的接收方法，使用锁保护
    async def _safe_receive(self):
        async with self.recv_lock:
            return await receive_message(self.websocket)

    async def stream_audio(
        self, 
        text_generator: AsyncGenerator[str, None], 
        output_callback: callable
    ):
        if not self.is_connected:
            await self.connect()
        
        if not self.session_id:
            await self.start_session()

        audio_buffer = bytearray()
        producer_task = None

        # 任务 1: 持续从生成器读取文本并发送
        async def producer():
            nonlocal producer_task
            producer_task = asyncio.current_task()
            try:
                async for chunk in text_generator:
                    if chunk.strip():
                        await self.send_text_chunk(chunk)
            except Exception as e:
                logger.error(f"Producer error: {e}")
            finally:
                # 文本生成结束，通知服务端会话结束
                await self.end_session()

        # 任务 2: 持续接收音频数据
        async def consumer():
            nonlocal audio_buffer
            try:
                while True:
                    # 检查是否已经收到 SessionFinished 信号（如果 producer 已经结束并触发了）
                    # 但我们需要通过消息来确认
                    
                    msg = await self._safe_receive()
                    
                    if msg.type == MsgType.FullServerResponse:
                        if msg.event == EventType.SessionFinished:
                            logger.debug("Received SessionFinished event.")
                            break
                        elif msg.event == EventType.TTSSentenceEnd:
                            # 可选：处理句子结束逻辑
                            pass
                    elif msg.type == MsgType.AudioOnlyServer:
                        if msg.payload:
                            audio_buffer.extend(msg.payload)
                            # 调用传入的回调函数，将音频块传递出去
                            await output_callback(msg.payload)
                    else:
                        logger.warning(f"Received unexpected message type: {msg.type}")
            except Exception as e:
                # 如果是连接关闭导致的错误，且 session 已结束，则忽略
                if self.session_id is None and "closed" in str(e).lower():
                    logger.debug("Consumer stopped because connection/session closed.")
                else:
                    logger.error(f"Consumer error: {e}")
                raise

        try:
            # 并发执行
            await asyncio.gather(producer(), consumer())
            logger.info(f"Stream complete. Total audio bytes: {len(audio_buffer)}")
            return bytes(audio_buffer)
        except Exception as e:
            logger.error(f"Streaming process failed: {e}")
            raise
        finally:
            # 确保锁被释放（async with 会自动处理，这里只是逻辑兜底）
            pass

# =======================================================
# 对外暴露的核心接口
# =======================================================

async def stream_tts_from_generator(
    text_generator: AsyncGenerator[str, None], 
    output_file_path: Optional[str] = None
) -> bytes:
    """
    [对外露接口]
    将 LLM 的文本流转换为语音流，并实时播放。
    """
    engine = TTSEngine()
    full_audio = bytearray()

    # --- 使用优化的FFmpeg进行流式播放 ---
    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = CONFIG["SAMPLE_RATE"]

    # 启动FFmpeg，注意添加了 -flush_packets 0 参数，可能有助于缓冲区管理
    # bufsize=0 设置管道为无缓冲
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', 'pipe:',
        '-f', 's16le',
        '-ar', str(RATE),
        '-ac', str(CHANNELS),
        '-flush_packets', '0', # 关键：减少输出缓冲
        'pipe:1'
    ]
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(os.devnull, 'w'), # 隐藏FFmpeg的错误输出
        bufsize=0 # 关键：设置为无缓冲
    )

    # 创建用于播放的队列和线程
    pcm_queue = queue.Queue()
    
    def playback_thread_func():
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=1024)
        while True:
            pcm_data = pcm_queue.get()
            if pcm_data is None:
                break
            if len(pcm_data) > 0:
                stream.write(pcm_data)
            pcm_queue.task_done()
        stream.stop_stream()
        stream.close()

    playback_thread = threading.Thread(target=playback_thread_func, daemon=True)
    playback_thread.start()

    def decode_and_queue_thread_func():
        chunk_size = 1024 * 2 # 1024 samples * 2 bytes per sample
        while True:
            pcm_chunk = ffmpeg_process.stdout.read(chunk_size)
            if not pcm_chunk:
                # 当FFmpeg的stdout关闭时，说明所有数据都已解码
                # 向播放队列发送结束信号
                pcm_queue.put(None)
                break
            if len(pcm_chunk) > 0:
                pcm_queue.put(pcm_chunk)

    decode_thread = threading.Thread(target=decode_and_queue_thread_func, daemon=True)
    decode_thread.start()

    async def play_audio_chunk(chunk: bytes):
        loop = asyncio.get_event_loop()
        # 将接收到的MP3块写入FFmpeg进程的stdin
        await loop.run_in_executor(None, lambda: ffmpeg_process.stdin.write(chunk))
        await loop.run_in_executor(None, lambda: ffmpeg_process.stdin.flush())
        full_audio.extend(chunk)

    try:
        await engine.connect()
        await engine.stream_audio(text_generator, play_audio_chunk)
        
        # 流程结束后，优雅地关闭FFmpeg
        # 1. 关闭stdin，通知FFmpeg输入结束
        ffmpeg_process.stdin.close()
        # 2. 等待FFmpeg进程自然退出，这会确保它处理完所有缓存
        ffmpeg_process.wait()
        # 3. 解码线程会因为stdout关闭而自动退出，并发送None给播放线程
        # 4. 等待播放线程播放完剩余音频后退出
        playback_thread.join()

        final_data = bytes(full_audio)
        
        if output_file_path:
            with open(output_file_path, "wb") as f:
                f.write(final_data)
            logger.info(f"Audio saved to {output_file_path}")
            
        return final_data
        
    except Exception as e:
        logger.error(f"TTS Streaming failed: {e}")
        raise
    finally:
        # 强制清理，以防万一
        try:
            if ffmpeg_process.poll() is None: # 如果进程还在运行
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
        
        p.terminate()
        await engine.close()

# =======================================================
# 测试模块
# =======================================================

async def mock_llm_generator():
    """模拟 LLM 流式输出"""
    # 使用一个较短的句子进行测试，更贴近真实场景
    test_text = "你好，这是一个测试。我正在模拟大语言模型的流式输出。你会听到声音随着文字的产生而逐渐播放出来。这是火山引擎 TTS 的流式合成功能。完整播放。"
    words = test_text.split(' ')
    for word in words:
        yield word + " "
        await asyncio.sleep(0.15) # 模拟生成延迟

async def run_test():
    print("--- 开始测试 LLM TTS 流式模块 (FFmpeg版 + 优化) ---")
    
    if CONFIG["APP_ID"] == "YOUR_APP_ID_HERE":
        print("错误：请先在代码顶部硬编码配置您的 APP_ID 和 ACCESS_TOKEN！")
        return

    output_filename = f"test_output_optimized.{CONFIG['OUTPUT_FORMAT']}"
    
    try:
        audio_data = await stream_tts_from_generator(
            text_generator=mock_llm_generator(),
            output_file_path=output_filename
        )
        
        print(f"✅ 测试成功！生成的音频大小：{len(audio_data)} 字节")
        print(f"📁 文件已保存至：{os.path.abspath(output_filename)}")
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())