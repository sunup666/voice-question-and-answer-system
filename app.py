import uuid
import time
import json
import random
import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms
import modelscope_studio.components.pro as pro
from config import DEFAULT_LOCALE, DEFAULT_SETTINGS, DEFAULT_THEME, DEFAULT_SUGGESTIONS, save_history, get_text, user_config, bot_config, welcome_config, markdown_config, MODEL_OPTIONS_MAP
from ui_components.logo import Logo
from ui_components.settings_header import SettingsHeader
from ui_components.thinking_button import ThinkingButton
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TextIteratorStreamer
from threading import Thread

# ---------- faster-whisper 快速语音识别（速度优化版）----------
from faster_whisper import WhisperModel
import numpy as np
import wave
import io
import base64
import os
import scipy.signal  # 仅用于重采样，放在顶部避免重复导入

# 自动设置 PyTorch 线程数（充分利用 CPU 核心）
cpu_count = os.cpu_count() or 4
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)

# ---------- TTS ----------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("警告: pyttsx3 未安装，朗读功能不可用。")

# ---------- faster-whisper 缓存 ----------
_whisper_cache = None

def get_whisper_model():
    global _whisper_cache
    if _whisper_cache is None:
        model_path = "D:/project/qwen3-chat/models/faster-whisper-small"
        if not os.path.exists(model_path):
            model_path = "small"
            print("本地 faster-whisper 不存在，使用 'small' 缓存模型")
        compute_type = "int8"
        device = "cpu"
        num_workers = cpu_count  # 根据 CPU 核心数自动设置
        print(f"加载 faster-whisper 模型：{model_path} (workers={num_workers}) ...")
        model = WhisperModel(model_path, device=device, compute_type=compute_type, num_workers=num_workers)
        _whisper_cache = model
        print("Whisper 模型加载完成")
    return _whisper_cache

def transcribe_wav_bytes(wav_bytes, task="transcribe"):
    """
    速度优先版转录（无初始提示，无时间戳，beam_size=1, 无 VAD）
    :param task: "transcribe" 或 "translate"
    """
    try:
        model = get_whisper_model()
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if n_channels > 1:
            audio_float32 = audio_float32.reshape(-1, n_channels).mean(axis=1)

        # 仅当采样率不是 16000 Hz 时才重采样（避免不必要的计算）
        if framerate != 16000:
            number_of_samples = int(len(audio_float32) * 16000 / framerate)
            audio_float32 = scipy.signal.resample(audio_float32, number_of_samples)

        # 速度优先配置：beam_size=1（贪心解码），无 VAD，无时间戳，无初始提示
        segments, info = model.transcribe(
            audio_float32,
            language=None,               # 自动检测语言
            task=task,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            word_timestamps=False,
            without_timestamps=True,     # 额外提速（faster-whisper 特有）
        )

        text = "".join(segment.text for segment in segments).strip()
        print(f"[faster-whisper] 识别结果（task={task}）: {text}")
        return text
    except Exception as e:
        print(f"[faster-whisper] 错误: {e}")
        import traceback
        traceback.print_exc()
        return ""  # 识别失败返回空字符串，避免将错误信息填入输入框

def transcribe_base64_audio(base64_str, task="transcribe"):
    if not base64_str:
        return ""
    try:
        wav_bytes = base64.b64decode(base64_str)
        return transcribe_wav_bytes(wav_bytes, task)
    except Exception as e:
        print(f"[Whisper] Base64错误: {e}")
        return ""

# ---------- LLM 模型（保持不变）----------
_model_cache = {}

def get_model(model_path):
    if model_path not in _model_cache:
        print(f"Loading model {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        _model_cache[model_path] = (tokenizer, model)
        print(f"Model {model_path} loaded.")
    return _model_cache[model_path]

def format_history(history, sys_prompt):
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    for item in history:
        if item["role"] == "user":
            messages.append({"role": "user", "content": item["content"]})
        elif item["role"] == "assistant":
            text_content = ""
            for content in item["content"]:
                if content.get("type") == "text":
                    text_content = content.get("content", "")
                    break
            messages.append({"role": "assistant", "content": text_content})
    return messages

def stream_generate(model, tokenizer, messages, enable_thinking):
    torch.manual_seed(int(time.time() * 1000) % 2**32)
    random.seed(time.time())

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.95,
        top_p=0.95,
        repetition_penalty=1.1,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    THINK_TAG_START = "<think>"
    THINK_TAG_END = "</think>"
    buffer = ""
    is_in_think = False

    for new_text in streamer:
        buffer += new_text
        while True:
            if not is_in_think:
                start_pos = buffer.find(THINK_TAG_START)
                if start_pos != -1:
                    before_think = buffer[:start_pos]
                    if before_think:
                        yield {"content": before_think, "reasoning_content": None}
                    buffer = buffer[start_pos + len(THINK_TAG_START):]
                    is_in_think = True
                else:
                    if buffer:
                        yield {"content": buffer, "reasoning_content": None}
                        buffer = ""
                    break
            else:
                end_pos = buffer.find(THINK_TAG_END)
                if end_pos != -1:
                    reasoning_part = buffer[:end_pos]
                    if reasoning_part:
                        yield {"content": None, "reasoning_content": reasoning_part}
                    buffer = buffer[end_pos + len(THINK_TAG_END):]
                    is_in_think = False
                else:
                    break

    if buffer:
        if is_in_think:
            yield {"content": None, "reasoning_content": buffer}
        else:
            yield {"content": buffer, "reasoning_content": None}
    thread.join()

# ---------- Gradio 事件类（保持不变）----------
class Gradio_Events:
    @staticmethod
    def submit(state_value):
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        settings = state_value["conversation_contexts"][state_value["conversation_id"]]["settings"]
        enable_thinking = state_value["conversation_contexts"][state_value["conversation_id"]]["enable_thinking"]
        model_path = settings.get("model")
        sys_prompt = settings.get("sys_prompt", "")

        messages = format_history(history, sys_prompt)

        history.append({
            "role": "assistant",
            "content": [],
            "key": str(uuid.uuid4()),
            "header": MODEL_OPTIONS_MAP.get(model_path, {}).get("label", None),
            "loading": True,
            "status": "pending"
        })
        yield {
            chatbot: gr.update(value=history),
            state: gr.update(value=state_value),
        }

        try:
            tokenizer, model = get_model(model_path)
            start_time = time.time()
            reasoning_content = ""
            answer_content = ""
            is_thinking = False
            is_answering = False
            contents = [None, None]

            for chunk in stream_generate(model, tokenizer, messages, enable_thinking):
                delta_reasoning = chunk.get("reasoning_content")
                delta_content = chunk.get("content")

                if delta_reasoning:
                    if not is_thinking:
                        contents[0] = {
                            "type": "tool",
                            "content": "",
                            "options": {
                                "title": get_text("Thinking...", "思考中..."),
                                "status": "pending"
                            },
                            "copyable": False,
                            "editable": False
                        }
                        is_thinking = True
                    reasoning_content += delta_reasoning

                if delta_content:
                    if not is_answering:
                        thought_cost_time = "{:.2f}".format(time.time() - start_time)
                        if contents[0]:
                            contents[0]["options"]["title"] = get_text(
                                f"End of Thought ({thought_cost_time}s)",
                                f"已深度思考 (用时{thought_cost_time}s)"
                            )
                            contents[0]["options"]["status"] = "done"
                        contents[1] = {
                            "type": "text",
                            "content": "",
                        }
                        is_answering = True
                    answer_content += delta_content

                if contents[0]:
                    contents[0]["content"] = reasoning_content
                if contents[1]:
                    contents[1]["content"] = answer_content
                history[-1]["content"] = [c for c in contents if c is not None]
                history[-1]["loading"] = False
                yield {
                    chatbot: gr.update(value=history),
                    state: gr.update(value=state_value)
                }

            history[-1]["status"] = "done"
            cost_time = "{:.2f}".format(time.time() - start_time)
            history[-1]["footer"] = get_text(f"{cost_time}s", f"用时{cost_time}s")
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value),
            }
        except Exception as e:
            print(f"model: {model_path} - Error: {e}")
            history[-1]["loading"] = False
            history[-1]["status"] = "done"
            history[-1]["content"] += [{
                "type": "text",
                "content": f'<span style="color: var(--color-red-500)">{str(e)}</span>'
            }]
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value)
            }
            raise e

    @staticmethod
    def add_message(input_value, settings_form_value, thinking_btn_state_value, state_value):
        if not state_value["conversation_id"]:
            random_id = str(uuid.uuid4())
            history = []
            state_value["conversation_id"] = random_id
            state_value["conversation_contexts"][state_value["conversation_id"]] = {"history": history}
            state_value["conversations"].append({"label": input_value, "key": random_id})

        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        state_value["conversation_contexts"][state_value["conversation_id"]] = {
            "history": history,
            "settings": settings_form_value,
            "enable_thinking": thinking_btn_state_value["enable_thinking"]
        }
        history.append({
            "role": "user",
            "content": input_value,
            "key": str(uuid.uuid4())
        })
        yield Gradio_Events.preprocess_submit(clear_input=True)(state_value)
        try:
            for chunk in Gradio_Events.submit(state_value):
                yield chunk
        except Exception as e:
            raise e
        finally:
            yield Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def preprocess_submit(clear_input=True):
        def preprocess_submit_handler(state_value):
            history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
            return {
                **( {
                    input: gr.update(value=None, loading=True) if clear_input else gr.update(loading=True),
                } if clear_input else {}),
                conversations: gr.update(
                    active_key=state_value["conversation_id"],
                    items=list(
                        map(lambda item: {
                            **item,
                            "disabled": True if item["key"] != state_value["conversation_id"] else False,
                        }, state_value["conversations"]))
                ),
                add_conversation_btn: gr.update(disabled=True),
                clear_btn: gr.update(disabled=True),
                conversation_delete_menu_item: gr.update(disabled=True),
                chatbot: gr.update(
                    value=history,
                    bot_config=bot_config(disabled_actions=['edit', 'retry', 'delete']),
                    user_config=user_config(disabled_actions=['edit', 'delete'])
                ),
                state: gr.update(value=state_value),
            }
        return preprocess_submit_handler

    @staticmethod
    def postprocess_submit(state_value):
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        return {
            input: gr.update(loading=False),
            conversation_delete_menu_item: gr.update(disabled=False),
            clear_btn: gr.update(disabled=False),
            conversations: gr.update(items=state_value["conversations"]),
            add_conversation_btn: gr.update(disabled=False),
            chatbot: gr.update(value=history, bot_config=bot_config(), user_config=user_config()),
            state: gr.update(value=state_value),
        }

    @staticmethod
    def cancel(state_value):
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history[-1]["loading"] = False
        history[-1]["status"] = "done"
        history[-1]["footer"] = get_text("Chat completion paused", "对话已暂停")
        return Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def delete_message(state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history = history[:index] + history[index + 1:]
        state_value["conversation_contexts"][state_value["conversation_id"]]["history"] = history
        return gr.update(value=state_value)

    @staticmethod
    def edit_message(state_value, chatbot_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history[index]["content"] = chatbot_value[index]["content"]
        if not history[index].get("edited"):
            history[index]["edited"] = True
            history[index]["footer"] = ((history[index]["footer"]) + " " if history[index].get("footer") else "") + get_text("Edited", "已编辑")
        return gr.update(value=state_value), gr.update(value=history)

    @staticmethod
    def regenerate_message(settings_form_value, thinking_btn_state_value, state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history = history[:index]
        state_value["conversation_contexts"][state_value["conversation_id"]] = {
            "history": history,
            "settings": settings_form_value,
            "enable_thinking": thinking_btn_state_value["enable_thinking"]
        }
        yield Gradio_Events.preprocess_submit()(state_value)
        try:
            for chunk in Gradio_Events.submit(state_value):
                yield chunk
        except Exception as e:
            raise e
        finally:
            yield Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def select_suggestion(input_value, e: gr.EventData):
        input_value = input_value[:-1] + e._data["payload"][0]
        return gr.update(value=input_value)

    @staticmethod
    def apply_prompt(e: gr.EventData):
        return gr.update(value=e._data["payload"][0]["value"]["description"])

    @staticmethod
    def new_chat(thinking_btn_state, state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_id"] = ""
        thinking_btn_state["enable_thinking"] = False
        return gr.update(active_key=state_value["conversation_id"]), gr.update(value=None), gr.update(value=DEFAULT_SETTINGS), gr.update(value=thinking_btn_state), gr.update(value=state_value)

    @staticmethod
    def select_conversation(thinking_btn_state_value, state_value, e: gr.EventData):
        active_key = e._data["payload"][0]
        if state_value["conversation_id"] == active_key or (active_key not in state_value["conversation_contexts"]):
            return gr.skip()
        state_value["conversation_id"] = active_key
        thinking_btn_state_value["enable_thinking"] = state_value["conversation_contexts"][active_key]["enable_thinking"]
        return gr.update(active_key=active_key), gr.update(value=state_value["conversation_contexts"][active_key]["history"]), gr.update(value=state_value["conversation_contexts"][active_key]["settings"]), gr.update(value=thinking_btn_state_value), gr.update(value=state_value)

    @staticmethod
    def click_conversation_menu(state_value, e: gr.EventData):
        conversation_id = e._data["payload"][0]["key"]
        operation = e._data["payload"][1]["key"]
        if operation == "delete":
            del state_value["conversation_contexts"][conversation_id]
            state_value["conversations"] = [item for item in state_value["conversations"] if item["key"] != conversation_id]
            if state_value["conversation_id"] == conversation_id:
                state_value["conversation_id"] = ""
                return gr.update(items=state_value["conversations"], active_key=state_value["conversation_id"]), gr.update(value=None), gr.update(value=state_value)
            else:
                return gr.update(items=state_value["conversations"]), gr.skip(), gr.update(value=state_value)
        return gr.skip()

    @staticmethod
    def toggle_settings_header(settings_header_state_value):
        settings_header_state_value["open"] = not settings_header_state_value["open"]
        return gr.update(value=settings_header_state_value)

    @staticmethod
    def clear_conversation_history(state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_contexts"][state_value["conversation_id"]]["history"] = []
        return gr.update(value=None), gr.update(value=state_value)

    @staticmethod
    def update_browser_state(state_value):
        return gr.update(value=dict(
            conversations=state_value["conversations"],
            conversation_contexts=state_value["conversation_contexts"]
        ))

    @staticmethod
    def apply_browser_state(browser_state_value, state_value):
        state_value["conversations"] = browser_state_value["conversations"]
        state_value["conversation_contexts"] = browser_state_value["conversation_contexts"]
        return gr.update(items=browser_state_value["conversations"]), gr.update(value=state_value)

    @staticmethod
    def read_last_assistant(state_value):
        conv_id = state_value.get("conversation_id")
        if not conv_id:
            return
        history = state_value["conversation_contexts"].get(conv_id, {}).get("history", [])
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content_parts = msg.get("content", [])
                text_to_speak = ""
                for part in content_parts:
                    if part.get("type") == "text":
                        text_to_speak = part.get("content", "")
                        break
                if text_to_speak:
                    speak_text(text_to_speak)
                else:
                    print("没有可朗读的文本")
                return
        print("未找到助手消息")

def speak_text(text: str):
    if not text or not text.strip():
        return
    if not TTS_AVAILABLE:
        print("pyttsx3 不可用")
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS] 朗读失败: {e}")

# ---------- CSS ----------
css = """
.gradio-container { padding: 0 !important; }
.gradio-container > main.fillable { padding: 0 !important; }
#chatbot { height: calc(100vh - 21px - 16px); max-height: 1500px; }
#chatbot .chatbot-conversations { height: 100vh; background-color: var(--ms-gr-ant-color-bg-layout); padding-left: 4px; padding-right: 4px; }
#chatbot .chatbot-conversations .chatbot-conversations-list { padding-left: 0; padding-right: 0; }
#chatbot .chatbot-chat { padding: 32px; padding-bottom: 0; height: 100%; }
@media (max-width: 768px) { #chatbot .chatbot-chat { padding: 0; } }
#chatbot .chatbot-chat .chatbot-chat-messages { flex: 1; }
#chatbot .setting-form-thinking-budget .ms-gr-ant-form-item-control-input-content { display: flex; flex-wrap: wrap; }
.voice-input-btn { margin-right: 5px; }
.read-btn { margin-right: 4px; }
.voice-input-btn.recording {
    background-color: #ff4444 !important;
    color: white !important;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}
"""

model_options_map_json = json.dumps(MODEL_OPTIONS_MAP)

# ---------- 录音 JS（不变）----------
recording_js = """
(function() {
    console.log('[录音] 初始化脚本 - 直接录制 WAV 格式');

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let audioContext = null;
    let mediaStream = null;

    async function startRecording(btn) {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(mediaStream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            audioChunks = [];
            
            processor.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                audioChunks.push(pcmData);
            };
            
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            window._recordingProcessor = processor;
            window._recordingSource = source;
            window._recordingContext = audioContext;
            window._recordingStream = mediaStream;
            
            isRecording = true;
            
            if (btn) {
                btn.classList.add('recording');
                btn.title = '点击停止录音';
            }
            console.log('[录音] 开始录音 (16kHz WAV)');
        } catch (error) {
            console.error('[录音] 失败:', error);
            alert('无法访问麦克风，请检查权限');
        }
    }

    function stopRecording(btn) {
        if (!isRecording) return;
        
        try {
            if (window._recordingProcessor) {
                window._recordingProcessor.disconnect();
            }
            if (window._recordingSource) {
                window._recordingSource.disconnect();
            }
            if (window._recordingContext) {
                window._recordingContext.close();
            }
            if (window._recordingStream) {
                window._recordingStream.getTracks().forEach(track => track.stop());
            }
            
            const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combined = new Int16Array(totalLength);
            let offset = 0;
            for (const chunk of audioChunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            
            const wavBuffer = encodeWAV(combined, 16000);
            const blob = new Blob([wavBuffer], { type: 'audio/wav' });
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                const hiddenInput = document.querySelector('#recording_data');
                if (hiddenInput) {
                    const textarea = hiddenInput.querySelector('textarea');
                    if (textarea) {
                        textarea.value = base64Audio;
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }
            };
            reader.readAsDataURL(blob);
            
            if (btn) {
                btn.classList.remove('recording');
                btn.title = '点击开始语音输入';
            }
            
            window._recordingProcessor = null;
            window._recordingSource = null;
            window._recordingContext = null;
            window._recordingStream = null;
            audioChunks = [];
            isRecording = false;
            
            console.log('[录音] 停止录音，已生成 WAV');
        } catch (error) {
            console.error('[录音] 停止失败:', error);
        }
    }

    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);
        
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            view.setInt16(offset, samples[i], true);
            offset += 2;
        }
        
        return buffer;
    }

    function writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    function findMicrophoneButton() {
        let btn = document.querySelector('.voice-input-btn');
        if (btn && btn.offsetHeight > 0) return btn;
        const allBtns = document.querySelectorAll('button');
        for (let b of allBtns) {
            if (b.innerHTML.includes('AudioOutlined') || b.querySelector('.anticon-audio')) {
                if (b.offsetHeight > 0) return b;
            }
        }
        return null;
    }

    let attempts = 0;
    const maxAttempts = 30;
    const interval = setInterval(() => {
        const btn = findMicrophoneButton();
        if (btn) {
            clearInterval(interval);
            console.log('[录音] 找到麦克风按钮');
            if (btn.hasAttribute('data-recording-inited')) {
                return;
            }
            btn.setAttribute('data-recording-inited', 'true');
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (isRecording) {
                    stopRecording(btn);
                } else {
                    startRecording(btn);
                }
            });
            console.log('[录音] 初始化完成');
        } else if (++attempts >= maxAttempts) {
            clearInterval(interval);
            console.error('[录音] 未找到麦克风按钮');
        }
    }, 500);
})();
"""

js = "function init() { window.MODEL_OPTIONS_MAP=" + model_options_map_json + ";" + recording_js + " }"

# ---------- 构建界面 ----------
with gr.Blocks(css=css, js=js, fill_width=True) as demo:
    state = gr.State({
        "conversation_contexts": {},
        "conversations": [],
        "conversation_id": "",
    })
    
    recording_data = gr.Textbox(visible=False, elem_id="recording_data")

    with ms.Application(), antdx.XProvider(theme=DEFAULT_THEME, locale=DEFAULT_LOCALE), ms.AutoLoading():
        with antd.Row(gutter=[20, 20], wrap=False, elem_id="chatbot"):
            with antd.Col(md=dict(flex="0 0 260px", span=24, order=0), span=0, order=1, elem_style=dict(width=0)):
                with ms.Div(elem_classes="chatbot-conversations"):
                    with antd.Flex(vertical=True, gap="small", elem_style=dict(height="100%")):
                        Logo()
                        with antd.Button(value=None, color="primary", variant="filled", block=True) as add_conversation_btn:
                            ms.Text(get_text("New Conversation", "新建对话"))
                            with ms.Slot("icon"):
                                antd.Icon("PlusOutlined")
                        with antdx.Conversations(elem_classes="chatbot-conversations-list") as conversations:
                            with ms.Slot('menu.items'):
                                with antd.Menu.Item(label="Delete", key="delete", danger=True) as conversation_delete_menu_item:
                                    with ms.Slot("icon"):
                                        antd.Icon("DeleteOutlined")
            with antd.Col(flex=1, elem_style=dict(height="100%")):
                with antd.Flex(vertical=True, gap="small", elem_classes="chatbot-chat"):
                    chatbot = pro.Chatbot(elem_classes="chatbot-chat-messages", height=0, markdown_config=markdown_config(), welcome_config=welcome_config(), user_config=user_config(), bot_config=bot_config())
                    with antdx.Suggestion(items=DEFAULT_SUGGESTIONS, should_trigger="""(e, { onTrigger, onKeyDown }) => {
                      switch(e.key) {
                        case '/': onTrigger(); break;
                        case 'ArrowRight': case 'ArrowLeft': case 'ArrowUp': case 'ArrowDown': break;
                        default: onTrigger(false);
                      }
                      onKeyDown(e)
                    }""") as suggestion:
                        with ms.Slot("children"):
                            with antdx.Sender(placeholder=get_text("Enter \"/\" to get suggestions", "输入 \"/\" 获取提示")) as input:
                                with ms.Slot("header"):
                                    settings_header_state, settings_form = SettingsHeader()
                                with ms.Slot("prefix"):
                                    with antd.Flex(gap=4, wrap=True, elem_style=dict(maxWidth='60vw')):
                                        # 原有按钮
                                        with antd.Button(value=None, type="text", elem_classes="voice-input-btn") as voice_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("AudioOutlined")
                                        with antd.Button(value=None, type="text", elem_classes="read-btn") as read_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("SoundOutlined")
                                        with antd.Button(value=None, type="text") as setting_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("SettingOutlined")
                                        with antd.Button(value=None, type="text") as clear_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("ClearOutlined")
                                        thinking_btn_state = ThinkingButton()
                                        
                                        # ---------- 仅保留翻译开关 ----------
                                        with ms.Div(elem_style=dict(marginLeft='auto', display='flex', gap='8px', alignItems='center')):
                                            translate_switch = antd.Switch(value=False, size="small")
                                            ms.Text(get_text("En→Zh", "中→英"), elem_style=dict(fontSize='12px'))

    if save_history:
        browser_state = gr.BrowserState(
            {"conversation_contexts": {}, "conversations": []},
            storage_key="qwen3_chat_demo_storage"
        )
        state.change(fn=Gradio_Events.update_browser_state, inputs=[state], outputs=[browser_state])
        demo.load(fn=Gradio_Events.apply_browser_state, inputs=[browser_state, state], outputs=[conversations, state])

    # 处理语音数据（仅翻译开关）
    def process_audio_data(base64_str, current_input, translate):
        if not base64_str:
            return gr.update(), gr.update()
        
        task = "translate" if translate else "transcribe"
        text = transcribe_base64_audio(base64_str, task=task)
        
        if text:
            new_input = (current_input or "") + text
            return gr.update(value=new_input), gr.update(value="")
        return gr.update(), gr.update(value="")
    
    recording_data.change(
        fn=process_audio_data,
        inputs=[recording_data, input, translate_switch],
        outputs=[input, recording_data]
    )

    read_btn.click(fn=Gradio_Events.read_last_assistant, inputs=[state], outputs=[])

    add_conversation_btn.click(fn=Gradio_Events.new_chat, inputs=[thinking_btn_state, state], outputs=[conversations, chatbot, settings_form, thinking_btn_state, state])
    conversations.active_change(fn=Gradio_Events.select_conversation, inputs=[thinking_btn_state, state], outputs=[conversations, chatbot, settings_form, thinking_btn_state, state])
    conversations.menu_click(fn=Gradio_Events.click_conversation_menu, inputs=[state], outputs=[conversations, chatbot, state])

    chatbot.welcome_prompt_select(fn=Gradio_Events.apply_prompt, outputs=[input])
    chatbot.delete(fn=Gradio_Events.delete_message, inputs=[state], outputs=[state])
    chatbot.edit(fn=Gradio_Events.edit_message, inputs=[state, chatbot], outputs=[state, chatbot])
    regenerating_event = chatbot.retry(fn=Gradio_Events.regenerate_message, inputs=[settings_form, thinking_btn_state, state], outputs=[input, clear_btn, conversation_delete_menu_item, add_conversation_btn, conversations, chatbot, state])

    submit_event = input.submit(fn=Gradio_Events.add_message, inputs=[input, settings_form, thinking_btn_state, state], outputs=[input, clear_btn, conversation_delete_menu_item, add_conversation_btn, conversations, chatbot, state])
    input.cancel(fn=Gradio_Events.cancel, inputs=[state], outputs=[input, conversation_delete_menu_item, clear_btn, conversations, add_conversation_btn, chatbot, state], cancels=[submit_event, regenerating_event], queue=False)

    setting_btn.click(fn=Gradio_Events.toggle_settings_header, inputs=[settings_header_state], outputs=[settings_header_state])
    clear_btn.click(fn=Gradio_Events.clear_conversation_history, inputs=[state], outputs=[chatbot, state])
    suggestion.select(fn=Gradio_Events.select_suggestion, inputs=[input], outputs=[input])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=100, max_size=100).launch(ssr_mode=False, max_threads=100)