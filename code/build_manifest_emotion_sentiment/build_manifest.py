import pandas as pd
from pathlib import Path

DATA_ROOT = Path("/data/wzw/egolink_race/data")

ANNOTATION = DATA_ROOT / "annotation" / "data1.xlsx"
VIDEO_ROOT = DATA_ROOT / "E3" / "E3"
AUDIO_ROOT = DATA_ROOT / "audio"
SUBTITLE_ROOT = DATA_ROOT / "subtext"

OUT_PATH = DATA_ROOT / "manifest"/"manifest.csv"

df = pd.read_excel(ANNOTATION)

rows = []
missing_video = 0
missing_audio = 0
missing_subtitle = 0

for _, row in df.iterrows():
    vid = str(row["Vid"])
    sub_id = str(row["sub_id"])

    video_path = VIDEO_ROOT / vid / f"{sub_id}.mp4"
    audio_path = AUDIO_ROOT / vid / f"{sub_id}.wav"
    subtitle_path = SUBTITLE_ROOT / vid / f"{sub_id}.vtt"

    if not video_path.exists():
        missing_video += 1
        continue

    if not audio_path.exists():
        missing_audio += 1
        continue

    if not subtitle_path.exists():
        missing_subtitle += 1
        continue

    rows.append({
        "Vid": vid,
        "sub_id": sub_id,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "subtitle_path": str(subtitle_path),
        "label": int(row["sentiment"]),
        "split": row["set"],
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(OUT_PATH, index=False)

print("manifest saved:", OUT_PATH)
print("usable samples:", len(manifest))
print("missing_video:", missing_video)
print("missing_audio:", missing_audio)
print("missing_subtitle:", missing_subtitle)
print(manifest["split"].value_counts())
print(manifest["label"].value_counts().sort_index())
"""这里建立只有在data1.csv中存在子视频，生成一个.csv文件。里面包含了vid、sub_id、video_path是该子视频路径、audio_path是该子音频路径、subtitle_path是该子视频的字母文件路径、label是情感状态标签、split是子视频是什么类别"""
