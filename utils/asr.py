"""
完整的vad模块
在 vad_asr.py 基础上，增加一个可注入的回调函数
"""

import os
import time
import threading
import queue
import urllib.request
import numpy as np
import torch
import sounddevice as sd
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

# --- VAD 模型本地缓存路径 ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)
VAD_CACHE_DIR = os.path.join(_PROJECT_DIR, "data", "vad_models")
os.makedirs(VAD_CACHE_DIR, exist_ok=True)

# Silero VAD 官方 ONNX 模型下载地址 (约 2MB)
SILERO_VAD_ONNX_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
# 备用镜像（Hugging Face，国内可能更快）
SILERO_VAD_ONNX_MIRROR_URL = (
    "https://huggingface.co/runanywhere/silero-vad-v5/resolve/main/silero_vad.onnx"
)
SILERO_VAD_ONNX_PATH = os.path.join(VAD_CACHE_DIR, "silero_vad.onnx")


def _download_file(url: str, dest: str) -> None:
    """下载文件到本地，支持超时和重定向。"""
    print(f"  下载中: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  完成 ({size_mb:.1f} MB) -> {dest}")
    except Exception as e:
        # 清理残留文件
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"下载失败: {e}")


def _make_onnx_wrapper(sess) -> "ONNXVADWrapper":
    """创建兼容 PyTorch VAD 调用的 ONNX 包装器。"""
    class ONNXVADWrapper:
        def __call__(self, audio_tensor, sr: int):
            audio_np = audio_tensor.cpu().numpy() if torch.is_tensor(audio_tensor) else audio_tensor
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            inp = {sess.get_inputs()[0].name: audio_np,
                   sess.get_inputs()[1].name: np.array([sr], dtype=np.int64)}
            out = sess.run(None, inp)
            return torch.tensor(out[0])
        def to(self, _device):
            return self
        def eval(self):
            return self
        def reset_states(self):
            pass
    return ONNXVADWrapper()


def _onnx_available() -> bool:
    """检查 onnxruntime 是否可用。"""
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def _load_vad_model_local(device: torch.device):
    """从本地缓存加载 VAD 模型，若不存在则下载后再加载。

    优先使用 ONNX 格式 (加载快，~2MB)，回退到原始 torch.hub 方式。
    """
    # 尝试本地缓存的 ONNX 模型
    if os.path.exists(SILERO_VAD_ONNX_PATH) and _onnx_available():
        print(f"  从本地加载 VAD 模型 (ONNX): {SILERO_VAD_ONNX_PATH}")
        try:
            import onnxruntime
            opts = onnxruntime.SessionOptions()
            opts.log_severity_level = 3
            sess = onnxruntime.InferenceSession(
                SILERO_VAD_ONNX_PATH, opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            print("  VAD 模型加载成功 (ONNX)")
            return _make_onnx_wrapper(sess)
        except Exception as e:
            print(f"  ONNX 加载失败 ({e})，尝试重新下载...")
            os.remove(SILERO_VAD_ONNX_PATH)

    # 首次运行或缓存失效: 尝试下载 ONNX 模型
    if not os.path.exists(SILERO_VAD_ONNX_PATH) and _onnx_available():
        print("  首次运行，下载 VAD 模型到本地缓存 (~2MB)...")
        downloaded = False
        for url in [SILERO_VAD_ONNX_URL, SILERO_VAD_ONNX_MIRROR_URL]:
            try:
                _download_file(url, SILERO_VAD_ONNX_PATH)
                downloaded = True
                break
            except Exception as e:
                print(f"  从 {url} 下载失败: {e}")
                if os.path.exists(SILERO_VAD_ONNX_PATH):
                    os.remove(SILERO_VAD_ONNX_PATH)

        if downloaded:
            try:
                import onnxruntime
                sess = onnxruntime.InferenceSession(SILERO_VAD_ONNX_PATH)
                print("  VAD 模型加载成功 (ONNX)")
                return _make_onnx_wrapper(sess)
            except Exception as e:
                print(f"  ONNX 加载失败: {e}")
                if os.path.exists(SILERO_VAD_ONNX_PATH):
                    os.remove(SILERO_VAD_ONNX_PATH)
        else:
            # 所有 URL 都失败，走 torch.hub 兜底
            print("  所有下载地址均失败，将使用 torch.hub 兜底。")

    if not _onnx_available():
        print("  onnxruntime 未安装，使用 torch.hub 原生加载 (首次需下载 ~95MB)")

    # 兜底: 返回 None，由调用者使用 torch.hub.load
    return None


class RealTimeASR:
    def __init__(self, on_recognized_callback=None):
        # 回调函数
        self.on_recognized_callback = on_recognized_callback

        # 初始化 FunASR 模型
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            quantize=True,
            disable_update=True,
            trust_remote_code=True,
        )

        # 设备一致性
        self.device = next(self.asr_model.model.parameters()).device

        # 初始化 VAD 模型 (带本地缓存)
        print("正在加载 VAD 模型 (Silero)...")
        self.vad_model = _load_vad_model_local(self.device)
        if self.vad_model is None:
            # 兜底: 使用 torch.hub
            print("  使用 torch.hub 加载 (首次可能需要下载)...")
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            self.get_speech_ts = utils[0]
        else:
            self.get_speech_ts = None  # ONNX 模式不使用

        self.vad_model.to(self.device)
        self.vad_model.eval()

        # 音频参数
        self.sample_rate = 16000
        self.frame_size = 512  
        
        # VAD 参数
        self.min_silence_duration_ms = 800
        self.min_speech_duration_ms = 250
        
        # 内部状态
        self.listening = True
        self.audio_queue = queue.Queue()
        self.speech_buffer = np.array([], dtype=np.float32)
        
        # VAD 状态变量
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_start_time = None

        # --- 新增：用于控制是否处理音频的事件 ---
        self._should_process = threading.Event()
        self._should_process.set() # 默认设置为True，即开始时允许处理

        # --- 表情与事件映射表 ---
        # SenseVoice 输出的标签 -> Emoji
        self.emoji_map = {
            # 情感 (Emotions)
            "<|HAPPY|>": "😄",
            "<|SAD|>": "😢",
            "<|ANGRY|>": "😡",
            "<|NEUTRAL|>": "", # 中性通常不显示表情，或者可以用 😐
            
            # 事件 (Events)
            "<|LAUGH|>": "🤣",   # 笑声
            "<|CRY|>": "😭",     # 哭声
            "<|APPLAUSE|>": "👏", # 掌声
            "<|COUGH|>": "🤧",   # 咳嗽
            "<|SUCTION|>": "😮", # 吸气/惊讶声
            
            # 语种标签 (通常在后处理中去除，这里以防万一也映射为空)
            "<|zh|>": "",
            "<|en|>": "",
            "<|ja|>": "",
            "<|ko|>": "",
        }

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())

    def reset_vad_state(self):
        if hasattr(self.vad_model, 'reset_states'):
            self.vad_model.reset_states()
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_start_time = None
        self.speech_buffer = np.array([], dtype=np.float32)

    # --- 新增：暂停处理函数 ---
    def pause_listening(self):
        """
        暂停ASR的语音处理，但保持麦克风流开启。
        这样可以避免因频繁开关音频流带来的性能损耗。
        """
        self._should_process.clear()
        logger.debug("[ASR] 监听已暂停，音频流保持开启。")

    # --- 新增：恢复处理函数 ---
    def resume_listening(self):
        """
        恢复ASR的语音处理。
        """
        self._should_process.set()
        logger.debug("[ASR] 监听已恢复。")
        # 可选：清空在暂停期间可能积压的音频队列
        self._clear_audio_queue()

    def _clear_audio_queue(self):
        """
        清空音频队列，丢弃在暂停期间收集的音频数据。
        """
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("[ASR] 已清空暂停期间的积压音频数据。")

    def process_vad_and_asr(self, audio_chunk):
        # --- 核心改动：在处理前检查是否允许处理 ---
        if not self._should_process.is_set():
            return # 如果不允许处理，则直接返回，不执行任何VAD或ASR逻辑

        chunk = audio_chunk.flatten().astype(np.float32)
        self.speech_buffer = np.concatenate([self.speech_buffer, chunk])

        audio_tensor = torch.from_numpy(chunk).to(self.device)
        
        with torch.no_grad():
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

        current_time = time.time()

        if speech_prob > 0.5:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time
        elif self.is_speaking:
            if self.last_speech_time:
                silence_duration = (current_time - self.last_speech_time) * 1000
                if silence_duration >= self.min_silence_duration_ms:
                    total_duration = (self.last_speech_time - self.speech_start_time) * 1000
                    if total_duration >= self.min_speech_duration_ms:
                        print("\n[语音结束] -> 识别中...")
                        self.run_asr_on_buffer()
                    self.reset_vad_state()

    def add_emojis_to_text(self, raw_text):
        """
        解析 SenseVoice 的原始标签并替换为 Emoji
        原始格式示例: <|HAPPY|>今天天气真好<|LAUGH|>
        """
        result_text = raw_text
        detected_emojis = []

        # 1. 提取所有标签并转换为 Emoji
        for tag, emoji in self.emoji_map.items():
            if tag in result_text:
                if emoji: # 如果映射了具体的 emoji
                    detected_emojis.append(emoji)
                # 从文本中移除原始标签 (如 <|HAPPY|>)
                result_text = result_text.replace(tag, "")
        
        # 2. 进行基础的标点后处理 (去除语种标签等)
        # 注意：rich_transcription_postprocess 主要处理标点和 ITN，也会去除部分标签
        # 但为了保留情感标签供我们转换，我们先手动替换情感标签，再调用后处理，
        # 或者直接用我们自己的逻辑清理。
        # 这里采用策略：先提取情感，清理标签，再调用官方 postprocess 处理标点。
        
        # 再次调用官方后处理以确保标点符号正确 (此时情感标签已被我们移除)
        final_text = rich_transcription_postprocess(result_text)
        
        # 3. 将检测到的表情添加到句首或句尾
        if detected_emojis:
            # 去重并保持顺序
            unique_emojis = list(dict.fromkeys(detected_emojis))
            emoji_str = "".join(unique_emojis)
            # 策略：放在句首，或者如果句子末尾是句号则放在句号前
            if final_text.endswith(("。", "！", "？", ".", "!", "?")):
                # 插在最后一个标点前
                final_text = final_text[:-1] + " " + emoji_str + final_text[-1]
            else:
                final_text = emoji_str + " " + final_text
                
        return final_text

    def run_asr_on_buffer(self):
        if len(self.speech_buffer) == 0:
            return

        try:
            audio_data = self.speech_buffer.copy()
            
            # 关键：generate 返回的 text 字段包含原始标签，如 <|HAPPY|>你好<|LAUGH|>
            res = self.asr_model.generate(
                input=audio_data, 
                cache={},
                is_final=True,
                use_itn=True,
                language="auto"
            )
            
            if res and len(res) > 0:
                raw_text = res[0].get("text", "")
                
                # 自定义处理：添加表情
                final_text = self.add_emojis_to_text(raw_text)
                
                if final_text.strip():
                    print(f">>> {final_text}")
                    if self.on_recognized_callback:
                        self.on_recognized_callback(final_text)
                else:
                    print(">>> [无有效内容]")
            else:
                print(">>> [ASR 无输出]")

        except Exception as e:
            print(f"[ERROR]: {e}")
        
        self.reset_vad_state()

    def start_listening(self):
        print("启动麦克风... (支持情感表情 😄😢😡🤣)")
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype=np.float32
        ):
            print("就绪。请带着感情说话...")
            while self.listening:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    # --- 核心改动：在获取队列数据后，仍需检查是否应处理 ---
                    # 这是为了处理在队列中有数据，但处理状态变为暂停的情况
                    # if not self._should_process.is_set():
                    #     continue # 这一行可以加也可以不加，取决于是否希望丢弃队列中已有的数据
                    self.process_vad_and_asr(audio_chunk)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.listening = False
                    break

# def main():
#     torch.set_num_threads(4)
#     asr_system = RealTimeASR()
#     try:
#         asr_system.start_listening()
#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()