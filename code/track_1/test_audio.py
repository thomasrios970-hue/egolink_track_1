"""
脚本作用：
临时测试本地 Qwen2.5-Omni 是否能直接理解 wav 音频内容，不经过 Whisper，并搭配一张随机视频帧。

执行逻辑：
1. 从 data/audio 中随机选择一个 wav 音频。
2. 从 data/processed/frames_16 中随机选择一张图片帧。
3. 加载本地 model/modelscope_model/Qwen2.5-Omni-7B。
4. 将 wav 音频和随机图片帧一起输入 Qwen2.5-Omni，并询问音频在说什么。
5. 打印音频 Vid_sub_id、音频路径、图片路径和模型回答。

运行示例：
CUDA_VISIBLE_DEVICES=4,5 python code/track_1/test.py --seed 42
或：
python code/track_1/test_audio.py --seed 45 --gpu 4,5
"""

"""需要的库统一在这里导入"""
import argparse
import contextlib
import os
import random
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--gpu", default="4,5", help="使用哪些 GPU，如 4 或 4,5；为空则不设置")
parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
args = parser.parse_args()
if args.gpu.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
rng = random.Random(args.seed)

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.utils import logging as transformers_logging
from qwen_omni_utils import process_mm_info

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

"""所有输入输出路径都从 config 已有路径拼接出来"""
audio_root = Path(config.PATH_TO_DATA_DIR) / "audio"
frame_root = Path(config.PATH_TO_PROCESSED_DIR) / "frames_16"
model_path = Path(config.PATH_TO_MODEL_DIR) / "modelscope_model" / "Qwen2.5-Omni-7B"


"""调用本地 Qwen2.5-Omni 分析音频，输入：音频路径和图片路径 -> 输出：模型回答字符串"""
def call_local_qwen_omni(audio_path, frame_path):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that understands audio."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "image", "image": str(frame_path)},
                {"type": "text", "text": "The image is only auxiliary. Focus on the audio and briefly describe what is being said. Use English."},
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    output_ids = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=args.max_new_tokens)
    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


"""代码块部分：随机选择音频和图片帧，并调用本地 Qwen2.5-Omni"""
audio_paths = sorted(audio_root.glob("*/*.wav"))
frame_paths = sorted(frame_root.glob("*/*/*.jpg")) + sorted(frame_root.glob("*/*/*.png"))
if not audio_paths:
    raise FileNotFoundError(f"no wav found under {audio_root}")
if not frame_paths:
    raise FileNotFoundError(f"no frame found under {frame_root}")
if not model_path.exists():
    raise FileNotFoundError(f"model not found: {model_path}")

audio_path = rng.choice(audio_paths)
frame_path = rng.choice(frame_paths)
Vid = audio_path.parent.name
sub_id = audio_path.stem

with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        enable_audio_output=False,
    )
    answer = call_local_qwen_omni(audio_path, frame_path)  # 输入：音频路径和图片路径；输出：模型直接听音频后的回答。

print("audio_test_result")
print(f"  audio_vid_sub_id: {Vid}_{sub_id}")
print(f"  audio_path: {audio_path}")
print(f"  frame_path: {frame_path}")
print(f"  model_answer: {answer}")
