"""
脚本作用：
临时测试“Whisper 音频转文本 + 图片”是否能让大模型理解音频内容。

执行逻辑：
1. 从 data/audio 中随机选择一个 wav 音频。
2. 优先从同一个 Vid/sub_id 的 processed frames 中随机选择一张图片；找不到则从全部 frames 中随机选一张。
3. 使用 Whisper 把音频转成文本，再把音频文本和图片一起输入大模型。
4. 打印音频 Vid_sub_id、音频路径、随机帧路径和模型回答。

运行示例：
python code/track_1/test.py --seed 42
"""

"""需要的库统一在这里导入"""
import argparse
import base64
import json
import os
import random
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError

import whisper

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

"""所有输入输出路径都从 config 已有路径拼接出来"""
audio_root = Path(config.PATH_TO_DATA_DIR) / "audio"
processed_root = Path(config.PATH_TO_PROCESSED_DIR)
audio_text_root = processed_root / "audio_to_text"
MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "gpt-5.5")
QWEN_API_URL = os.environ.get("QWEN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--model_name", default="small", help="Whisper 模型名称")
parser.add_argument("--gpu", type=int, default=4, help="Whisper 使用的 GPU，-1 表示 CPU")
args = parser.parse_args()
rng = random.Random(args.seed)


"""将文件转成 base64，输入：文件路径 -> 输出：base64 字符串"""
def encode_file(path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


"""按 Vid/sub_id 找一张帧图，输入：Vid、sub_id -> 输出：图片路径或 None"""
def pick_frame(Vid, sub_id):
    frame_paths = []
    for frame_dir_name in ["frames_16", "frames_8", "frames_4", "frames_32"]:
        frame_dir = processed_root / frame_dir_name / str(Vid) / str(sub_id)
        frame_paths.extend(sorted(frame_dir.glob("*.jpg")))
        frame_paths.extend(sorted(frame_dir.glob("*.png")))
    if not frame_paths:
        frame_paths = list(processed_root.glob("frames_*/*/*/*.jpg"))
    return rng.choice(frame_paths) if frame_paths else None


"""用 Whisper 转写音频并保存，输入：音频路径、Vid、sub_id -> 输出：转写文本 str"""
def transcribe_audio(audio_path, Vid, sub_id):
    save_path = audio_text_root / str(Vid) / f"{sub_id}.txt"
    if save_path.exists():
        return save_path.read_text(encoding="utf-8").strip()
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"
    whisper_model = whisper.load_model(
        name=args.model_name,
        device=device,
        download_root=str(Path(config.PATH_TO_MODEL_DIR) / "huggingface_model" / "whisper"),
    )
    result = whisper_model.transcribe(str(audio_path), verbose=False, fp16=False)
    audio_text = " ".join(segment.get("text", "") for segment in result.get("segments", [])).strip()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(audio_text, encoding="utf-8")
    return audio_text


"""调用大模型判断音频文本内容，输入：音频转写文本、图片路径 -> 输出：模型回答字符串"""
def call_api(audio_text, frame_path):
    content = [
        {"type": "text", "text": "The following text is transcribed from an audio clip. Briefly answer what the audio is describing or saying. Ignore the image content; the image is only attached to test multimodal input. Use English.\n\nAudio transcript:\n" + audio_text},
    ]
    if frame_path is not None:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_file(frame_path)}"}})

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_completion_tokens": 256,
    }
    headers = {"Content-Type": "application/json"}
    if QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_API_KEY}"

    req = request.Request(
        QWEN_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        error_text = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API request failed: HTTP {e.code}, {error_text}") from e

    return str(result.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


"""代码块部分：随机选择音频和帧，并调用 API"""
audio_paths = sorted(audio_root.glob("*/*.wav"))
if not audio_paths:
    raise FileNotFoundError(f"no wav found under {audio_root}")

audio_path = rng.choice(audio_paths)
Vid = audio_path.parent.name
sub_id = audio_path.stem
frame_path = pick_frame(Vid, sub_id)  # 输入：Vid、sub_id；输出：随机帧路径或 None。
audio_text = transcribe_audio(audio_path, Vid, sub_id)  # 输入：音频路径、Vid、sub_id；输出：Whisper 转写文本。
answer = call_api(audio_text, frame_path)  # 输入：音频转写文本、图片路径；输出：模型回答字符串。

print(f"audio_vid_sub_id: {Vid}_{sub_id}")
print(f"audio_path: {audio_path}")
print(f"frame_path: {frame_path}")
print(f"audio_text: {audio_text}")
print(f"model_answer: {answer}")
