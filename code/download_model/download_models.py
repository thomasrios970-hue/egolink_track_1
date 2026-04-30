from huggingface_hub import snapshot_download
#模型下载工具
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
SAVE_ROOT = Path(config.PATH_TO_HUGGINGFACE_MODEL)
#模型下载位置
audio_models={
    ################## CHINESE ######################
    "TencentGameMate/chinese-hubert-base": "chinese-hubert-base",
    "TencentGameMate/chinese-hubert-large": "chinese-hubert-large",
    "TencentGameMate/chinese-wav2vec2-base": "chinese-wav2vec2-base",
    "TencentGameMate/chinese-wav2vec2-large": "chinese-wav2vec2-large",
}

txt_models={
    "hfl/chinese-macbert-large": "chinese-macbert-large",

    # 多语强模型，推荐优先
    "FacebookAI/xlm-roberta-large": "xlm-roberta-large",
    # 更大，但很吃显存
    "facebook/xlm-roberta-xl": "xlm-roberta-xl",
    "facebook/xlm-roberta-xxl": "xlm-roberta-xxl",
}

visual_models={
    "openai/clip-vit-large-patch14": "clip-vit-large-patch14",

    "openai/clip-vit-large-patch14-336": "clip-vit-large-patch14-336",
    #这个image_size最好336，即--image_size 336
}

models = {
    "openai/clip-vit-large-patch14": "clip-vit-large-patch14",
    "TencentGameMate/chinese-hubert-large": "chinese-hubert-large",
    "hfl/chinese-macbert-large": "chinese-macbert-large",
}

from typing import Dict
def download_models(models:Dict):
    for repo_id, local_name in models.items():
        #repo_id是hugging_face上的模型名,local_name是本地保存的文件名
        print(f"Downloading {repo_id}")
        local_dir=SAVE_ROOT/local_name
        if (local_dir/"config.json").exists():
            print(f"{local_name}已存在")
        else:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"{local_name}已下载完毕")        
if __name__=="__main__":
    download_models(models|audio_models|txt_models|visual_models)
