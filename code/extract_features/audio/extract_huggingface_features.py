"""
从音频 manifest 中批量提取 HuggingFace 音频模型特征。

执行逻辑：
1. 从 download_models.py 的 audio_models 中读取当前脚本支持的音频模型名称。
2. 根据 --model_name 选择单个模型，或在 --model_name all 时依次处理所有支持的音频模型。
3. 对每个模型检查本地模型目录；如果缺少 config.json，则自动调用下载函数下载对应模型。
4. 读取 config.py 中配置的音频 manifest，manifest 需要包含 Vid、sub_id 和 audio_path 字段。
5. 对每条音频先检查目标特征文件是否已经存在；若所需 feature_level 全部存在，则跳过该音频。
6. 读取音频文件，双声道音频会先转为单声道，过短音频会跳过。
7. 使用 HuggingFace feature extractor 转成模型输入；音频过长时按 10 秒一段切分，最后一段不足 10 秒时补 0。
8. 使用模型提取 hidden states，将最后 4 层相加作为帧级音频特征，并按真实音频长度裁掉补 0 产生的多余帧。
9. 如果 feature_level 为 UTTERANCE，则对时间维度取平均并保存整段音频特征；如果为 FRAME，则保存帧级特征；如果为 all，则两者都保存。

运行示例：
python code/extract_features/audio/extract_huggingface_features.py \
  --feature_level all \
  --model_name all \
  --gpu 4
"""

from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


import config
from download_model.download_models import download_models, models,audio_models

import math
import torch
#创建参数解析器
import argparse

import soundfile as sf
import pandas as pd
import numpy as np

FEATURE_LEVEL=["UTTERANCE","FRAME"]
MODEL_NAME=list(audio_models.values())
parse=argparse.ArgumentParser()
def build_parser():
    parse.add_argument("--feature_level",
                       choices=FEATURE_LEVEL+["all"],
                       default="all",
                       help="特征级别，UTTERANCE 保存整段音频级特征，FRAME保存每一个时间片的特征",
                       )
    parse.add_argument("--model_name",
                       choices=MODEL_NAME+["all"],
                       default="chinese-hubert-large",
                       help="使用的 HuggingFace 模型名称",)
    parse.add_argument("--gpu",
                       default=4,
                       type=int,
                       help="使用的cpu卡的序列")
build_parser()
args=parse.parse_args()


"""如果音频太长则切分数据"""
def split_into_batch(input_value,maxlen=16000*10):
	if len(input_value[0])<=maxlen: #[0]表示input_values的第一个音频，这里是一个个处理
		return input_value
	
	bs,wavlen=input_value.shape
	assert bs==1
	tgtlen=math.ceil(wavlen/maxlen)*maxlen #将tgtlen向上补成maxlen的整数倍，比如12s就变成20s
	batches=torch.zeros(1,tgtlen)
	batches[:,:wavlen]=input_value #batches的前wavlen个采样点被真实样本数据覆盖
	batches=batches.view(-1,maxlen) #如(1,50000)->(1,64000)->(4,16000)
	return batches

target_model=MODEL_NAME if args.model_name=="all" else [args.model_name]
for model_name in target_model:
    """读取模型并创建音频处理器对象"""
    model_file=Path(config.PATH_TO_HUGGINGFACE_MODEL)/f"{model_name}"   

    """如果模型不存在则自动下载模型"""
    if not (model_file/"config.json").exists():
        models_to_download = {
        repo_id: local_name
        for repo_id, local_name in audio_models.items()
            if local_name == model_name
        }#筛选出来我们要下载的模型
        if len(models_to_download) == 0:
            raise ValueError(f"找不到模型 {model_name} 对应的 HuggingFace repo")
        download_models(models_to_download)


    model=AutoModel.from_pretrained(model_file)
    feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(model_file)

    """将模型放入gpu"""
    if args.gpu != -1:
        device = torch.device(f"cuda:{args.gpu}")
        model.to(device)
    model.eval() #进入推理模式

    """读取音频并提取特征"""
    manifest_path=Path(config.PATH_TO_AUDIO_MANIFEST)
    df=pd.read_csv(manifest_path)

    for idx,row in df.iterrows():
        Vid=row["Vid"]
        sub_id=row["sub_id"]
        audio_path=Path(row["audio_path"])

        #如果已经存在文件了就跳过
        target_levels = FEATURE_LEVEL if args.feature_level == "all" else [args.feature_level]
        save_paths = [
            Path(config.PATH_TO_AUDIO_FEATURE_DIR) / f"{model_name}" / level / f"{Vid}" / f"{sub_id}.npy"
            for level in target_levels
        ]

        if all(path.exists() for path in save_paths):
            print(f"({idx+1}/{len(df)})的{target_levels}特征已存在")
            continue

        print(f"Processing {audio_path}...({idx+1}/{len(df)})")

        """读取音频"""
        sample,sr=sf.read(audio_path)
        if sample.ndim > 1:
            sample = sample.mean(axis=1)
        #如果是双声道就先转单声道
        
        if len(sample) < 400:
            print("skip too short:", audio_path)
            continue
            #如果音频太短就跳过
        else:
            with torch.no_grad():
                """提取特征"""
                input_values=feature_extractor(sample,sampling_rate=sr,return_tensors="pt").input_values

                real_wavlen=input_values.shape[1] #真实音频长度，补0之前

                input_values=split_into_batch(input_values,maxlen=16000*10 ) #如果音频太长则切分数据
                """将音频放入gpu"""
                if args.gpu != -1:
                    input_values=input_values.to(device)
                    

                hidden_states=model(input_values,output_hidden_states=True).hidden_states

                layers_ids=[-1,-2,-3,-4] #提取最后4层的特征
                feature=torch.stack(hidden_states)[layers_ids].sum(dim=0)
                
                bsize,segnum,featdim=feature.shape
                feature=feature.view(-1,featdim) #将原来可能切分的数据拼接到一起
                valid_len = model._get_feat_extract_output_lengths(
                    torch.tensor([real_wavlen])
                ).item() 
                #计算有多少帧
                feature = feature[:valid_len]
                #把后面多余的帧去除

                feature=feature.detach().squeeze().cpu().numpy()
            """保存特征"""
            def save_feature(feature_level):
                try:
                    save_path=Path(config.PATH_TO_AUDIO_FEATURE_DIR)/f"{model_name}"/f"{feature_level}"/f"{Vid}"/f"{sub_id}.npy"
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
