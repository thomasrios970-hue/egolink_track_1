"""
脚本作用：
从全部 E3 子视频中提取 16k 单声道 wav 音频。

执行逻辑：
1. 遍历 Path(config.PATH_TO_DATA_DIR) / "E3" / "E3" 下所有 .mp4 视频。
2. 对每个视频按 Vid/sub_id 生成音频到 Path(config.PATH_TO_DATA_DIR) / "audio" / Vid / f"{sub_id}.wav"。
3. 如果音频已经存在则跳过，否则用 ffmpeg 提取 wav。

运行示例：
python code/extract_audio/extract_audio.py
"""

"""需要的库统一在这里导入"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

"""所有输入输出路径都从 config 已有路径拼接出来"""
video_root = Path(config.PATH_TO_DATA_DIR) / "E3" / "E3"
audio_root = Path(config.PATH_TO_DATA_DIR) / "audio"

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--progress_interval", type=int, default=500, help="每处理多少个视频打印一次进度")
args = parser.parse_args()

"""用 ffmpeg 从视频提取 wav 音频，输入：视频路径、音频保存路径 -> 输出：是否成功 bool"""
def extract_wav(video_path, audio_path):
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


"""文件检索与计数统一用这段代码"""
video_suffixes = [".mp4"]
video_count = len(list(video_root.rglob("*.mp4")))
file_idx = 0
success = 0
skipped = 0
failed = 0

for video_path in video_root.rglob("*"):
    if video_path.suffix not in video_suffixes:
        continue

    Vid = video_path.parent.name
    sub_id = video_path.stem
    file_idx += 1
    save_path = audio_root / Vid / f"{sub_id}.wav"

    if save_path.exists():
        skipped += 1
        continue

    print(f"正在提取第{file_idx}/{video_count}个音频")
    ok = extract_wav(video_path, save_path)  # 输入：视频路径、音频保存路径；输出：是否成功 bool。
    if ok:
        success += 1
    else:
        failed += 1

    if file_idx % args.progress_interval == 0:
        print(f"进度 {file_idx}/{video_count}, success={success}, skipped={skipped}, failed={failed}")

print("Done")
print(f"total={video_count}")
print(f"success={success}")
print(f"skipped={skipped}")
print(f"failed={failed}")
