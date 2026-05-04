"""
下载 Qwen2.5-Omni-7B 到本地模型目录。

脚本作用：
1. 使用 ModelScope 下载 Qwen2.5-Omni-7B，适合国内网络环境。
2. 默认保存到 Path(config.PATH_TO_HUGGINGFACE_MODEL) / "Qwen2.5-Omni-7B"。
3. 下载前检查磁盘空间，下载后检查关键文件是否存在。
4. 最后默认用 GPU 加载 Qwen2.5-Omni Thinker，并跑一条真实文本生成检查，确认模型能产出结果。

运行示例：
python code/download_model/download_qwen_omni.py

如果缺少 modelscope，请先运行：
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U modelscope
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_LOCAL_NAME = "Qwen2.5-Omni-7B"
DEFAULT_MIN_FREE_GB = 35
SUPPORTED_GPUS = [4, 5]
DEFAULT_GPU = 4

REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="使用 ModelScope 下载 Qwen2.5-Omni-7B 到本地模型目录。",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="ModelScope 模型 ID。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.PATH_TO_HUGGINGFACE_MODEL) / DEFAULT_LOCAL_NAME,
        help="模型保存目录。",
    )
    parser.add_argument(
        "--min-free-gb",
        type=positive_int,
        default=DEFAULT_MIN_FREE_GB,
        help="下载前要求的最小可用磁盘空间 GB。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=SUPPORTED_GPUS,
        default=DEFAULT_GPU,
        help="验证加载模型时使用的 GPU 卡号，只允许 4 或 5。",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="跳过模型加载验证，只检查文件是否完整。",
    )
    return parser


def check_disk_space(path, min_free_gb):
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1024 ** 3
    print(f"disk_free_gb: {free_gb:.1f}")
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"磁盘空间不足：剩余 {free_gb:.1f}GB，需要至少 {min_free_gb}GB"
        )


def download_model(model_id, output_dir):
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "缺少 modelscope，请先运行："
            "python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U modelscope"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(
        model_id,
        local_dir=str(output_dir),
    )
    print("downloaded_to:", model_dir)


def verify_files(output_dir):
    missing = [
        file_name
        for file_name in REQUIRED_FILES
        if not (output_dir / file_name).exists()
    ]
    if missing:
        raise RuntimeError(f"模型文件不完整，缺少：{missing}")

    safetensors = sorted(output_dir.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError("没有找到 .safetensors 权重文件")

    print(f"required_files_ok: {len(REQUIRED_FILES)}")
    print(f"safetensors_files: {len(safetensors)}")


def verify_model(output_dir, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
    except ImportError as e:
        raise RuntimeError(
            "缺少 Qwen2.5-Omni 推理依赖，请确认 transformers 版本支持 Qwen2.5-Omni。"
            "建议运行: python -m pip install -U 'transformers>=4.52.0'"
        ) from e

    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(output_dir),
        trust_remote_code=True,
    )
    print("processor_ok")

    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(output_dir),
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(output_dir),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen. Answer briefly.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Return exactly: Answer: A",
                }
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(get_model_input_device(model))
    output_ids = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
    )
    output_ids = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, output_ids)
    ]
    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if not text:
        raise RuntimeError("模型已加载，但生成结果为空")
    print("model_generate_ok:", text)


def get_model_input_device(model):
    for module_name in ("thinker", "model"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        try:
            return next(module.parameters()).device
        except StopIteration:
            continue
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("model_id:", args.model_id)
    print("output_dir:", args.output_dir)
    print("min_free_gb:", args.min_free_gb)
    print("gpu:", args.gpu)

    check_disk_space(args.output_dir.parent, args.min_free_gb)
    download_model(args.model_id, args.output_dir)
    verify_files(args.output_dir)

    if not args.skip_model_check:
        verify_model(args.output_dir, args.gpu)

    print("Qwen2.5-Omni model is ready.")


if __name__ == "__main__":
    main()
