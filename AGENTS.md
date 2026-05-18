# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 并且你要追寻一下这种格式：
# 脚本文件

```
你是一个追求功能完整但又讨厌程序冗杂的工程师，你在写代码的时候总是会遵守以下规范，代码行数越少越好，如果要保存文件请支持断点重续，如果要调用api要设置三次错误重试，然后超时就跳过后面再重试，其它格式如下："
"""脚本作用，如：
使用 Whisper 为音频生成按对话内容对齐的 VTT 字幕文件。
如需要调用API则根据要调用的大模型从环境变量中取出，例如QWEN大模型就是QWEN_API_KEY和QWEN_API_URL

执行逻辑：
如：
1. 从命令行读取 Whisper 模型名称和 GPU 编号，加载对应模型。
2. 遍历 `config.PATH_TO_FULL_DATA` 目录下的 `.wav` 音频文件。
3. 对每个音频文件执行转写，获取 Whisper 识别出的时间片段。

运行示例：
    如：python code/splice_audio.py --model_name small --gpu 4 --export_model json
"""

"""需要的库统一在这里导入"""
import whisper
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
#如果config在上上层目录就这样
import config
#所有的输入输出路径都放在config上或由config文件配合字符拼接出路径，不要放在参数上，然后尽量不要添加config里面的路径，能拼就拼
import argparse
import json
MODEL_NAME=["small","tiny","medium"]
"""如果全部模型都要跑一遍则这样操作后参数解析器后面+"all"，或者从其它文件中导入（如果有专门的模型下载脚本的话，如
from download_model import download_huggingface_models, huggingface_model
MODEL_NAME=list(huggingface_model.audio_models.values())）
之后遍历操作在download_models.py上有显示

"""参数解析器统一在这里设置和添加，参数尽量的少，能完成任务就行,否则运行代码会很长"""
"""大模型常见参数有--temperature：温度越低模型倾向于选择概率最高的答案，输出越稳定、保守；--gpu：选择在哪几个gpu上跑代码
--modelname：选用哪个模型"""
parser = argparse.ArgumentParser()
def build_parse():
    parser.add_argument(
        "--model_name",
        choice=MODEL_NAME+["all"]
        default="small",
        help="指定模型的大小/名称，如small,tiny等"
    )
build_parse()
args = parser.parse_args()

#自定义函数统一放这里，并表明用处，自定义函数不要那么多保险工程，越短越好，保留关键部分就行；且功能要完善，能够增加代码灵活度的才要放在这里否则不要放，要有一定复杂度的地方代码采用自定义函数，要有复用价值；并且每个自定义函数都要写出它的输入是什么，输出是什么，有什么作用，没有输入输出就写无输入，无输出；之后调用自定义函数的时候就在旁边使用#注释写输入和输出/结果
"""将秒变成VTT的格式，输入：秒->输出：vtt格式的秒"""
def format_vtt(seconds): 
    hours = int(seconds // 3600)
    minutes = int((seconds) % 3600 / 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"

"""用whisper的时间戳对齐之后以vtt格式导出,输入：音频路径、保存路径、和被whisper_model.transcribe转写返回的字典->结果：以vtt的格式导出对齐后的时间戳和文本"""
def export_vtt(file_path,save_path,result):
    #result为whisper_model.transcribe转写返回的字典
    #file_path被转写的音频路径
    segments = result.get("segments", [])
    lines = ["WEBVTT", ""]

    with open(file_path.with_suffix(".json"), "r", encoding="utf-8") as f:
        conversations = json.load(f)
        for idx, segment in enumerate(segments):
            if idx >= len(conversations):
                break

            text = segment.get("text", "")
            if text == "":
                continue

            start = format_vtt(segment.get("start", 0.0))
            end = format_vtt(segment.get("end", 0.0))
            lines.append(f"{start} --> {end}")
            lines.append(str(conversations[idx].get("content") or ""))
            lines.append("")

        vtt_text = "\n".join(lines)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(vtt_text, encoding="utf8")

#代码块部分，代码块部分越短越好，但是不是说全部都要用自定义函数，只有自定义函数比较复杂的时候才要求用自定义函数，要求在完整的实现功能前提下只保留关键部分的代码
whisper_model = whisper.load_model(
    name=args.model_name,
    device=f"cuda:{args.gpu}",
    download_root=str(Path(config.PATH_TO_EGOLINK_MODEL) / "whisper")
)

"""文件检索与计数统一用这段代码"""
data_path = Path(config.PATH_TO_FULL_DATA)
audio_suffiexs = [".wav"]
audio_count = len(list(data_path.rglob("*.wav")))
file_idx = 0

for file_path in data_path.rglob("*"):
    if file_path.suffix not in audio_suffiexs:
        continue

    Vid = file_path.parent.parent.name
    sub_id = file_path.stem
    file_idx += 1
    save_path = Path(config.PATH_TO_PROCESSED) / "splice_audio"/f"{args.export_model}"/Vid / sub_id / f"{sub_id}.{args.export_model}"
    if save_path.exists():
        print(f"{args.export_model}第{file_idx}/{audio_count}个文件已经存在")
        continue
    print(f"正在提取第{file_idx}/{audio_count}个文件")
    

    result = whisper_model.transcribe(
        str(file_path),
        verbose=False,
        fp16=False
    )

    save_path = Path(config.PATH_TO_PROCESSED) / "splice_audio"/f"{args.export_model}"/Vid / sub_id / f"{sub_id}.{args.export_model}"
    if args.export_model=="vtt":
        export_vtt(file_path,save_path,result)
    elif args.export_model=="json":
        export_json(file_path,save_path,result)
"
```

# config文件

```
import os
"""输入路径"""
PATH_TO_MODEL_DIR="/data/wzw/egolink_race/model"
PATH_TO_AUDIO_MANIFEST_DIR="/data/wzw/egolink_race/data/manifest"



"""输出路径"""
PATH_TO_PROCESSED_DIR="/data/wzw/egolink_race/data/processed"
PATH_TO_FEATURE_DIR="/data/wzw/egolink_race/feature"
```

# 下模型脚本

```
"""
脚本作用：
从 Hugging Face 下载指定的预训练模型，并保存到本地模型目录中。

执行逻辑：
1. 从 `config.PATH_TO_MODEL_DIR` 读取模型保存根目录。
2. 在 `huggingface_model` 字典中按模型类型组织需要下载的模型，包括音频模型、文本模型和视觉模型。
3. 依次遍历每个模型类型下的 Hugging Face 仓库名 `repo_id` 和本地保存名 `local_name`。
4. 将模型下载到 `{PATH_TO_MODEL_DIR}/huggingface_model/{local_name}`。
5. 如果目标目录下已经存在 `config.json`，则认为模型已下载，直接跳过。
6. 如果模型不存在，则调用 `snapshot_download` 下载完整模型文件。

运行示例：
    python code/download_model/download_models.py
"""
from huggingface_hub import snapshot_download as snapshot_download_huggingface
from modelscope import snapshot_download as snapshot_download_modelscope
#模型下载工具
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
SAVE_ROOT = Path(config.PATH_TO_MODEL_DIR)
#模型下载位置
#模型列表
huggingface_model={
    "audio_models":{
        "TencentGameMate/chinese-hubert-base": "chinese-hubert-base",
    },

    "txt_models":{
        "hfl/chinese-macbert-large": "chinese-macbert-large",
    },

    "visual_models":{
        "openai/clip-vit-large-patch14": "clip-vit-large-patch14",
    },
}
"""
遍历时这样写
for model_type, models in huggingface_model.items():
    print(model_type)

    for repo_id, local_name in models.items():
        print(repo_id, local_name)
"""
modelscope_model={
    "llm_model":{
        "Qwen/Qwen2.5-Omni-7B": "Qwen2.5-Omni-7B",
    },
}

from typing import Dict
def download_huggingface_models(huggingface_model:Dict):
    for _, models in huggingface_model.items():
        for repo_id, local_name in models.items():
            #repo_id是hugging_face上的模型名,local_name是本地保存的文件名
            print(f"Downloading {repo_id}")
            download_path=SAVE_ROOT/"huggingface_model"/local_name
            if (download_path/"config.json").exists():
                print(f"{local_name}已存在")
            else:
                snapshot_download_huggingface(
                    repo_id=repo_id,
                    local_dir=download_path,
                    local_dir_use_symlinks=False,
                )
                print(f"{local_name}已下载完毕")    

def download_modelscope_model(modelscope_model:Dict):
    for _,models in modelscope_model.items():
        for repo_id,local_name in models.items():
            print(f"Downloading {repo_id}")
        download_path=SAVE_ROOT/"modelscope_model"/local_name
        if (download_path/"config.json").exists():
                print(f"{local_name}已存在")
        else:
            snapshot_download_modelscope(
		    repo_id=repo_id,
            local_dir=download_path,
            )

if __name__=="__main__":
    download_huggingface_models(huggingface_model)
    download_modelscope_model(modelscope_model)
```

