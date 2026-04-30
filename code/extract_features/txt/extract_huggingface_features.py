"""
从字幕 manifest 中批量提取 HuggingFace 文本模型特征。

执行逻辑：
1. 从 download_models.py 的 txt_models 中读取当前脚本支持的文本模型名称。
2. 根据 --model_name 选择单个模型，或在 --model_name all 时依次处理所有支持的文本模型。
3. 对每个模型检查本地模型目录；如果缺少 config.json，则自动调用下载函数下载对应模型。
4. 读取 config.py 中配置的 manifest，manifest 需要包含 Vid、sub_id 和 subtitle_path 字段。
5. 对每条字幕先检查目标特征文件是否已经存在；若所需 feature_level 全部存在，则跳过该字幕。
6. 读取 WebVTT 字幕文件，过滤 WEBVTT 标记、时间轴和序号行，拼接为纯文本。
7. 使用 HuggingFace tokenizer 转成模型输入，并使用文本模型提取 hidden states。
8. 将最后 4 层 hidden states 相加作为 token-level 文本特征。
9. 如果 feature_level 为 UTTERANCE，则保存 [CLS] 位置的整段文本特征；如果为 FRAME，则保存 token-level 特征；如果为 all，则两者都保存。

运行示例：
python code/extract_features/txt/extract_huggingface_features.py \
  --feature_level all \
  --model_name all \
  --gpu 4
"""

from transformers import AutoModel
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


import config
from download_model.download_models import download_models, models,txt_models

import re
import torch
#创建参数解析器
import argparse

import pandas as pd
import numpy as np

FEATURE_LEVEL=["UTTERANCE","FRAME"]
MODEL_NAME=list(txt_models.values())
parse=argparse.ArgumentParser()
def build_parser():
    parse.add_argument("--feature_level",
                       choices=FEATURE_LEVEL+["all"],
                       default="all",
                       help="特征级别，UTTERANCE 保存整段文本级特征，FRAME保存每一个token的特征",
                       )
    parse.add_argument("--model_name",
                       choices=MODEL_NAME+["all"],
                       default="chinese-macbert-large",
                       help="使用的 HuggingFace 模型名称",)
    parse.add_argument("--gpu",
                       default=4,
                       type=int,
                       help="使用的cpu卡的序列")
    parse.add_argument("--max_length",
                       default=256,
                       type=int,
                       help="tokenizer的最大长度")
build_parser()
args=parse.parse_args()


"""读取字幕文本"""
def read_vtt(path):
    text=Path(path).read_text(encoding="utf-8",errors="ignore")
    lines=[]
    for line in text.splitlines():
        line=line.strip()
        if not line:
            continue
        if line=="WEBVTT":
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$",line):
            continue
        lines.append(line)
    return " ".join(lines)

target_model=MODEL_NAME if args.model_name=="all" else [args.model_name]
for model_name in target_model:
    """读取模型并创建文本处理器对象"""
    model_file=Path(config.PATH_TO_HUGGINGFACE_MODEL)/f"{model_name}"   

    """如果模型不存在则自动下载模型"""
    if not (model_file/"config.json").exists():
        models_to_download = {
        repo_id: local_name
        for repo_id, local_name in txt_models.items()
            if local_name == model_name
        }#筛选出来我们要下载的模型
        if len(models_to_download) == 0:
            raise ValueError(f"找不到模型 {model_name} 对应的 HuggingFace repo")
        download_models(models_to_download)


    model=AutoModel.from_pretrained(model_file)
    tokenizer=AutoTokenizer.from_pretrained(model_file)

    """将模型放入gpu"""
    if args.gpu != -1:
        device = torch.device(f"cuda:{args.gpu}")
        model.to(device)
    model.eval() #进入推理模式

    """读取字幕并提取特征"""
    manifest_path=Path(config.PATH_TO_AUDIO_MANIFEST)
    df=pd.read_csv(manifest_path)

    for idx,row in df.iterrows():
        Vid=row["Vid"]
        sub_id=row["sub_id"]
        subtitle_path=Path(row["subtitle_path"])

        #如果已经存在文件了就跳过
        target_levels = FEATURE_LEVEL if args.feature_level == "all" else [args.feature_level]
        save_paths = [
            Path(config.PATH_TO_TXT_FEATURE_DIR) / f"{model_name}" / level / f"{Vid}" / f"{sub_id}.npy"
            for level in target_levels
        ]

        if all(path.exists() for path in save_paths):
            print(f"({idx+1}/{len(df)})的{target_levels}特征已存在")
            continue

        print(f"Processing {subtitle_path}...({idx+1}/{len(df)})")

        """读取字幕"""
        text=read_vtt(subtitle_path)
        if not text:
            text="[PAD]"
        
        with torch.no_grad():
            """提取特征"""
            inputs=tokenizer(text,max_length=args.max_length,truncation=True,padding=True,return_tensors="pt")

            """将文本放入gpu"""
            if args.gpu != -1:
                inputs={key:value.to(device) for key,value in inputs.items()}
                

            hidden_states=model(**inputs,output_hidden_states=True).hidden_states

            layers_ids=[-1,-2,-3,-4] #提取最后4层的特征
            feature=torch.stack(hidden_states)[layers_ids].sum(dim=0)
            
            feature=feature.squeeze(0)

            feature=feature.detach().squeeze().cpu().numpy()
        """保存特征"""
        def save_feature(feature_level):
            try:
                save_path=Path(config.PATH_TO_TXT_FEATURE_DIR)/f"{model_name}"/f"{feature_level}"/f"{Vid}"/f"{sub_id}.npy"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if save_path.exists():
                    print(f"{save_path}已存在")
                    return
                if feature_level=="UTTERANCE":
                    feature_UTTERANCE=np.array(feature).squeeze()
                    if len(feature_UTTERANCE.shape)!=1:
                        feature_UTTERANCE=feature_UTTERANCE[0]
                    np.save(save_path,feature_UTTERANCE)
                else:
                    np.save(save_path,feature)
            except Exception as e:
                print(f"出现错误：{e}")
        if args.feature_level!="all":
            save_feature(args.feature_level)
        else:
            for feature_level in FEATURE_LEVEL:
                save_feature(feature_level)
