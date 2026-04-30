"""
从视频 manifest 中批量提取 HuggingFace 视觉模型特征。

执行逻辑：
1. 从 download_models.py 的 visual_models 中读取当前脚本支持的视觉模型名称。
2. 根据 --model_name 选择单个模型，或在 --model_name all 时依次处理所有支持的视觉模型。
3. 对每个模型检查本地模型目录；如果缺少 config.json，则自动调用下载函数下载对应模型。
4. 读取 config.py 中配置的 manifest，manifest 需要包含 Vid、sub_id 和 video_path 字段。
5. 对每条视频先检查目标特征文件是否已经存在；若所需 feature_level 全部存在，则跳过该视频。
6. 读取视频文件，并在整段视频中均匀抽取指定数量的帧。
7. 将抽到的帧 resize 到指定尺寸，并按 CLIP 的均值和方差做归一化预处理。
8. 使用 HuggingFace CLIPVisionModel 提取每帧视觉特征。
9. 如果 feature_level 为 UTTERANCE，则对帧维度取平均并保存整段视频特征；如果为 FRAME，则保存 frame-level 特征；如果为 all，则两者都保存。

运行示例：
python code/extract_features/visual/extract_huggingface_features.py \
  --feature_level all \
  --model_name all \
  --gpu 4
"""

from transformers import CLIPVisionModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


import config
from download_model.download_models import download_models, models,visual_models

import torch
#创建参数解析器
import argparse

import cv2
import pandas as pd
import numpy as np

FEATURE_LEVEL=["UTTERANCE","FRAME"]
MODEL_NAME=list(visual_models.values())
CLIP_MEAN_VALUES=(0.48145466,0.4578275,0.40821073)
CLIP_STD_VALUES=(0.26862954,0.26130258,0.27577711)
parse=argparse.ArgumentParser()
def build_parser():
    parse.add_argument("--feature_level",
                       choices=FEATURE_LEVEL+["all"],
                       default="all",
                       help="特征级别，UTTERANCE 保存整段视频级特征，FRAME保存每一帧的特征",
                       )
    parse.add_argument("--model_name",
                       choices=MODEL_NAME+["all"],
                       default="clip-vit-large-patch14",
                       help="使用的 HuggingFace 模型名称",)
    parse.add_argument("--gpu",
                       default=4,
                       type=int,
                       help="使用的cpu卡的序列")
    parse.add_argument("--num_frames",
                       default=8,
                       type=int,
                       help="每个视频均匀抽取的帧数")
    parse.add_argument("--image_size",
                       default=224,
                       type=int,
                       help="送入CLIP前的正方形图像尺寸")
build_parser()
args=parse.parse_args()


"""从视频中均匀抽帧"""
def sample_frames(video_path,num_frames,image_size):
    cap=cv2.VideoCapture(str(video_path))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total<=0:
        cap.release()
        raise ValueError("cannot read video frames")

    indices=np.linspace(0,total-1,num_frames).astype(int)
    frames=[]
    clip_mean=np.array(CLIP_MEAN_VALUES,dtype=np.float32)
    clip_std=np.array(CLIP_STD_VALUES,dtype=np.float32)

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
        ok,frame=cap.read()

        if not ok:
            continue

        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame=cv2.resize(frame,(image_size,image_size),interpolation=cv2.INTER_AREA)
        frame=frame.astype(np.float32)/255.0
        frame=(frame-clip_mean)/clip_std
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("no valid frames")

    frames=np.stack(frames,axis=0)
    frames=np.transpose(frames,(0,3,1,2))
    return torch.from_numpy(frames)

target_model=MODEL_NAME if args.model_name=="all" else [args.model_name]
for model_name in target_model:
    """读取模型并创建视觉处理器对象"""
    

    model_file=Path(config.PATH_TO_HUGGINGFACE_MODEL)/f"{model_name}"   
    """如果模型不存在则自动下载模型"""
    if not (model_file/"config.json").exists():
        models_to_download = {
        repo_id: local_name
        for repo_id, local_name in visual_models.items()
            if local_name == model_name
        }#筛选出来我们要下载的模型
        if len(models_to_download) == 0:
            raise ValueError(f"找不到模型 {model_name} 对应的 HuggingFace repo")
        download_models(models_to_download)


    model=CLIPVisionModel.from_pretrained(model_file)

    """将模型放入gpu"""
    if args.gpu != -1:
        device = torch.device(f"cuda:{args.gpu}")
        model.to(device)
    model.eval() #进入推理模式

    """读取视频并提取特征"""
    manifest_path=Path(config.PATH_TO_AUDIO_MANIFEST)
    df=pd.read_csv(manifest_path)

    for idx,row in df.iterrows():
        Vid=row["Vid"]
        sub_id=row["sub_id"]
        video_path=Path(row["video_path"])

        #如果已经存在文件了就跳过
        target_levels = FEATURE_LEVEL if args.feature_level == "all" else [args.feature_level]
        save_paths = [
            Path(config.PATH_TO_VISUAL_FEATURE_DIR) / f"{model_name}" / level / f"{Vid}" / f"{sub_id}.npy"
            for level in target_levels
        ]

        if all(path.exists() for path in save_paths):
            print(f"({idx+1}/{len(df)})的{target_levels}特征已存在")
            continue

        print(f"Processing {video_path}...({idx+1}/{len(df)})")

        """读取视频"""
        image_size = 336 if model_name=="clip-vit-large-patch14-336" else args.image_size
        try:
            pixel_values=sample_frames(video_path,args.num_frames,image_size)
        except Exception as e:
            print(f"skip invalid video: {video_path}, {e}")
            continue
        
        with torch.no_grad():
            """提取特征"""
            if args.gpu != -1:
                pixel_values=pixel_values.to(device)
                

            outputs=model(pixel_values=pixel_values)

            feature=outputs.pooler_output

            feature=feature.detach().squeeze().cpu().numpy()
        """保存特征"""
        def save_feature(feature_level):
            try:
                save_path=Path(config.PATH_TO_VISUAL_FEATURE_DIR)/f"{model_name}"/f"{feature_level}"/f"{Vid}"/f"{sub_id}.npy"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if save_path.exists():
                    print(f"{save_path}已存在")
                    return
                if feature_level=="UTTERANCE":
                    feature_UTTERANCE=np.array(feature).squeeze()
                    if len(feature_UTTERANCE.shape)!=1:
                        feature_UTTERANCE=feature_UTTERANCE.mean(axis=0)
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
