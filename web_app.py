"""
FastAPI Web Application for Real-Time Voice Assistant
Provides a real-time web UI via WebSocket for the ASR → LLM → TTS pipeline.

Usage:
    python web_app.py

Then open http://localhost:8000 in your browser.
"""

import asyncio
import io
import json
import os
import re
import sys
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- Ensure project root is in path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- Import existing modules ---
from utils.asr import RealTimeASR
from core.agent import generate_response
from utils.TTS.tts import stream_tts_from_generator
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage multiple WebSocket connections and broadcast messages."""

    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict):
        """Send JSON message to all connected clients."""
        stale = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


manager = ConnectionManager()


# ============================================================================
# Voice Pipeline Bridge (ASR → LLM → TTS → WebSocket)
# ============================================================================

class VoicePipeline:
    """Bridges the voice pipeline events to WebSocket clients.

    Runs ASR in a background thread, forwards recognition results to LLM,
    then TTS, broadcasting every stage via WebSocket.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.asr: RealTimeASR | None = None
        self.running = False
        self._processing = False
        self._thread: threading.Thread | None = None

    # -- Broadcasting (thread-safe) --

    def _emit(self, msg: dict):
        """Schedule a WebSocket broadcast on the main event loop."""
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(msg), self.loop)
        except RuntimeError:
            pass  # Event loop not ready yet

    # -- ASR callback (called from ASR thread) --

    def _on_asr_complete(self, text: str):
        """Called by RealTimeASR when a sentence is recognized."""
        if self._processing:
            return
        if not text or "[无有效内容]" in text or "[ASR 无输出]" in text:
            return

        self._processing = True

        # Detect emotion emoji at start of text (ASR already embeds them)
        if text and len(text) > 0:
            first_char = text[0]
            if ord(first_char) > 0x2000:  # Basic emoji range check
                self._emit({"type": "emotion", "data": first_char})

        self._emit({"type": "asr_done", "data": ""})
        self._process_and_respond(text)

    # -- Core processing pipeline --

    def _process_and_respond(self, text: str):
        """LLM → TTS pipeline running in a daemon thread."""
        self._emit({"type": "user_message", "data": text})
        self._emit({"type": "status", "data": "thinking"})

        def worker():
            worker_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(worker_loop)

            async def task():
                try:
                    # 1. LLM
                    config = {"configurable": {"thread_id": "web_conversation_001"}}
                    reply = generate_response(text, config=config)
                    self._emit({"type": "ai_message", "data": reply})

                    # 2. TTS
                    self._emit({"type": "status", "data": "speaking"})
                    self._emit({"type": "tts", "data": "start"})

                    if self.asr:
                        self.asr.pause_listening()
                    try:
                        await stream_tts_from_generator(
                            text_generator=self._text_streamer(reply),
                        )
                    finally:
                        if self.asr:
                            self.asr.resume_listening()

                    self._emit({"type": "tts", "data": "end"})
                    self._emit({"type": "status", "data": "listening"})

                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    self._emit({"type": "error", "data": str(e)})
                    self._emit({"type": "status", "data": "listening"})
                finally:
                    self._processing = False

            try:
                worker_loop.run_until_complete(task())
            finally:
                worker_loop.close()

        threading.Thread(target=worker, daemon=True).start()

    # -- TTS text streaming --

    @staticmethod
    async def _text_streamer(text: str) -> AsyncGenerator[str, None]:
        """Split text into sentences for streaming TTS."""
        sentences = re.split(r'([。！？.!?])', text)
        for i in range(0, len(sentences) - 1, 2):
            s = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if s.strip():
                yield s
                await asyncio.sleep(0.01)

    # -- Public control API --

    def start(self):
        """Start the voice pipeline (ASR listening)."""
        if self.running:
            return
        try:
            self.asr = RealTimeASR(on_recognized_callback=self._on_asr_complete)
            self.running = True
            self._emit({"type": "status", "data": "listening"})
            self._thread = threading.Thread(
                target=self.asr.start_listening, daemon=True,
            )
            self._thread.start()
            logger.info("Voice pipeline started")
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self._emit({"type": "error", "data": f"启动失败: {e}"})
            self.running = False

    def stop(self):
        """Stop the voice pipeline."""
        if self.asr:
            self.asr.listening = False
        self.running = False
        self._processing = False
        self._emit({"type": "status", "data": "stopped"})
        logger.info("Voice pipeline stopped")

    def chat(self, text: str):
        """Process a text message (typed input) through LLM + TTS."""
        if not text.strip():
            return
        if self._processing:
            self._emit({"type": "error", "data": "正在处理上一轮输入，请稍候"})
            return
        self._processing = True
        self._process_and_respond(text)


# ============================================================================
# FastAPI Application
# ============================================================================

pipeline: VoicePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    loop = asyncio.get_event_loop()
    pipeline = VoicePipeline(loop)
    logger.info("Web server started at http://localhost:8000")
    yield
    if pipeline:
        pipeline.stop()
    logger.info("Web server stopped")


app = FastAPI(title="Real-Time Voice Assistant", lifespan=lifespan)

web_dir = os.path.join(current_dir, "web")


# -- WebSocket endpoint --

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Send initial state
    if pipeline and pipeline.running:
        await ws.send_json({"type": "status", "data": "listening"})
    else:
        await ws.send_json({"type": "status", "data": "stopped"})

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "")

            if action == "start":
                if pipeline:
                    pipeline.start()
            elif action == "stop":
                if pipeline:
                    pipeline.stop()
            elif action == "chat":
                text = msg.get("message", "")
                if pipeline:
                    pipeline.chat(text)
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(ws)


# -- REST health endpoint --

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline_running": pipeline.running if pipeline else False,
    }


# Serve static web files (must be last — routes above take precedence)
if os.path.isdir(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # Fix Windows console encoding for emoji
    if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'gb2312'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("=" * 55)
    print("  [MIC] Real-Time Voice Assistant - Web UI")
    print()
    print("  Open http://localhost:8000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 55)
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
