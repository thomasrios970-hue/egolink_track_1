"""
脚本作用：
从 Hugging Face和modelscope下载指定的预训练模型，并保存到本地模型目录中。

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
        # 多模态本地大模型，支持视频、音频的输入
        "Qwen/Qwen2.5-Omni-7B": "Qwen2.5-Omni-7B",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen3-Omni-30B-A3B-Thinking": "Qwen3-Omni-30B-A3B-Thinking"
    },
    "audio_models":{
        "Qwen/Qwen3-ForcedAligner-0.6B": "Qwen3-ForcedAligner-0.6B", #对齐字幕用
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

