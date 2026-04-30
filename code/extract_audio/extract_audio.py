import subprocess
from pathlib import Path

import pandas as pd

ANNOTATION = Path("/data/wzw/egolink_race/data/annotation/data1.xlsx")
#视频信息路径
VIDEO_ROOT = Path("/data/wzw/egolink_race/data/E3/E3")
#视频路径
AUDIO_ROOT = Path("/data/wzw/egolink_race/data/audio")
#导出音频路径
df = pd.read_excel(ANNOTATION)

total = len(df)
missing = 0
skipped = 0
success = 0
failed = 0

for idx, row in df.iterrows():
    vid = str(row["Vid"])
    sub_id = str(row["sub_id"])

    video_path = VIDEO_ROOT / vid / f"{sub_id}.mp4"
    audio_path = AUDIO_ROOT / vid / f"{sub_id}.wav"

    if not video_path.exists():
        missing += 1
        continue

    if audio_path.exists():
        skipped += 1
        continue

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        success += 1
    else:
        failed += 1

    if (idx + 1) % 500 == 0:
        print(
            f"[{idx + 1}/{total}] "
            f"success={success}, skipped={skipped}, "
            f"missing={missing}, failed={failed}"
        )

print("Done")
print(f"total={total}")
print(f"success={success}")
print(f"skipped={skipped}")
print(f"missing={missing}")
print(f"failed={failed}")
