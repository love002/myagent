# RealTimeVoiceAssistant - 实时语音助手

一款基于大语言模型的实时语音对话助手，集成语音识别（ASR）、自然语言处理（LLM）和语音合成（TTS）能力，支持数据库自然语言查询功能。

## 功能特性

- **实时语音识别** - 基于 FunASR (SenseVoice) 和 Silero VAD 实现低延迟语音转文字
- **智能对话代理** - 基于 LangGraph 构建，支持多轮对话记忆和工具调用
- **流式语音合成** - 字节跳动火山引擎 TTS，实时流式输出音频
- **自然语言数据库查询** - 通过 SQL Agent 用自然语言查询业务数据
- **对话记忆持久化** - 基于 SQLite 的 LangGraph Checkpoint 存储
- **情感语音识别** - SenseVoice 情感检测，识别结果带表情符号

## 项目结构

```
RealTimeVoiceAssistant/
├── main.py                      # 应用入口，ASR-LLM-TTS 流程编排
├── core/                        # 核心代理模块
│   ├── agent.py                 # LangGraph 对话代理
│   ├── prompt.py                # 系统提示词配置
│   └── tool.py                  # Agent 工具注册（SQL Agent）
├── utils/                       # 工具模块
│   ├── asr.py                   # 语音识别（FunASR + Silero VAD）
│   ├── config.py                # 全局配置
│   ├── llms.py                  # LLM 工厂（支持 Qwen/OpenAI/Ollama）
│   ├── logger.py                # 日志管理
│   ├── memory_sqlite.py         # LangGraph SQLite 持久化
│   ├── sql_agent.py             # 自然语言 SQL 查询代理
│   └── TTS/                     # 语音合成模块
│       ├── protocols.py         # TTS 协议消息定义
│       └── tts.py               # 字节跳动 TTS 引擎
├── data/                        # 数据目录
│   ├── app.db                   # 业务数据库（用户、产品表）
│   └── memory.db                # 对话记忆数据库
└── logfile/                     # 日志文件目录
```

## 技术栈

| 类别 | 技术 |
|------|------|
| **语音识别 (ASR)** | FunASR (SenseVoiceSmall), Silero VAD, PyAudio |
| **大语言模型 (LLM)** | 阿里云通义千问 (Qwen), OpenAI, Ollama |
| **Agent 框架** | LangChain / LangGraph |
| **语音合成 (TTS)** | 字节跳动火山引擎 (双向 WebSocket API) |
| **数据库** | SQLite |
| **日志** | concurrent-log-handler |
| **深度学习框架** | PyTorch |

## 核心模块

### main.py - 主入口

应用主循环，负责 ASR → LLM → TTS 流程的编排：

```
用户说话 → ASR 语音识别 → LLM 生成回复 → TTS 语音播放 → 循环
```

**关键函数：**
- `main()` - 启动 RealTimeASR 监听
- `on_asr_complete()` - ASR 识别完成回调，触发 LLM 处理
- `get_llm_response()` - 调用 LangGraph Agent 获取回复（带记忆）
- `handle_speech_response()` - 调用 TTS 播放回复音频
- `text_to_speech_streamer()` - 将文本分句后流式发送给 TTS

### core/agent.py - 对话代理

基于 LangGraph 的 AI 对话代理，支持工具调用和对话记忆持久化。

**特性：**
- 使用 Qwen (通义千问) 作为默认模型
- 集成 SqliteSaver 实现对话记忆
- 支持 SQL Agent 工具调用

### core/tool.py - Agent 工具

为 Agent 注册可用工具，当前支持：

- `sql_agent` - 自然语言查询业务数据库

### utils/asr.py - 语音识别

实时语音识别模块，核心类 `RealTimeASR`：

- **VAD 检测** - 使用 Silero VAD 检测语音活动
- **语音识别** - 使用 SenseVoiceSmall 模型
- **情感检测** - 识别结果包含情感标签（开心、悲伤、笑声等）
- **暂停/恢复** - TTS 播放时暂停 ASR，避免音频反馈

**情感标签映射：**
```
<|HAPPY|>  → 😊
<|LAUGH|>  → 😂
<|SAD|>    → 😢
<|ANGRY|>  → 😠
<|Neutral|> → 💬
```

### utils/TTS/tts.py - 语音合成

字节跳动火山引擎 TTS，核心类 `TTSEngine`：

- **双向 WebSocket** - 低延迟流式交互
- **流式播放** - 边接收音频边播放
- **音频解码** - FFmpeg 解码 MP3 为 PCM
- **实时播放** - PyAudio 流式输出

### utils/sql_agent.py - SQL 查询代理

基于 LangChain 的自然语言数据库查询，支持 `users` 和 `products` 两张表。

### utils/llms.py - LLM 工厂

统一 LLM 接口，支持多后端：Qwen (默认)、OpenAI、Ollama。

## 配置说明

### 环境变量 (.env)

```bash
DASHSCOPE_API_KEY=        # 阿里云百炼 API Key（用于 Qwen）
DEEPSEEK_API_KEY=         # DeepSeek API Key（未使用）
TAVILY_API_KEY=           # Tavily 搜索（未使用）
```

### 配置文件 (utils/config.py)

```python
LOG_FILE = "logfile/app.log"      # 日志文件路径
LLM_TYPE = "qwen"                 # 默认 LLM 类型
MEMORY_DB_PATH = "data/memory.db" # 对话记忆数据库
SQL_AGENT_DB_PATH = "data/app.db" # 业务数据库
```

### TTS 配置 (utils/TTS/tts.py)

TTS 模块使用字节跳动火山引擎，需要配置有效的 `APP_ID` 和 `ACCESS_TOKEN`。

## 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖：** langchain, langgraph, funasr, silero-vad, pyaudio, python-dotenv, websockets, torch

## 使用方法

```bash
python main.py
```

程序启动后，ASR 开始监听麦克风。用户说话后进行语音识别、LLM 对话、TTS 播放，持续循环直到用户中断（Ctrl+C）。

## 数据库表结构

### users 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | TEXT | 姓名 |
| email | TEXT | 邮箱 |
| age | INTEGER | 年龄 |
| city | TEXT | 城市 |

### products 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | TEXT | 产品名称 |
| category | TEXT | 分类 |
| price | REAL | 价格 |
| stock | INTEGER | 库存 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           main.py                                    │
│                                                                      │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    │
│  │  RealTimeASR │───►│ on_asr_complete │───►│ get_llm_response │    │
│  │   (麦克风)    │    │    (回调)       │    │   (Agent + 记忆)  │    │
│  └──────────────┘    └─────────────────┘    └────────┬─────────┘    │
│                                                      │               │
│                                                      ▼               │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    │
│  │  TTSEngine   │◄───│ handle_speech   │◄───│  Agent Response  │    │
│  │  (火山引擎)   │    │   _response     │    │                  │    │
│  └──────────────┘    └─────────────────┘    └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 注意事项

1. **API Key** - 使用前需在 `.env` 中配置 `DASHSCOPE_API_KEY`
2. **TTS 凭证** - `utils/TTS/tts.py` 中需配置有效的火山引擎 APP_ID 和 ACCESS_TOKEN
3. **音频设备** - 确保系统有可用的麦克风和扬声器
4. **FFmpeg** - TTS 音频解码需要安装 FFmpeg 并添加到 PATH
5. **模型下载** - 首次运行时会自动下载 ASR 模型（SenseVoiceSmall, Silero VAD）
