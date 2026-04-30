"""
构建 emotion 分类任务的 manifest。

该脚本读取 data/annotation/data2.xlsx，其中 emotion 列是类似 happy、
angry、surprised 的离散情绪类别。每一行人物情绪标注都会保留为一个
样本；如果同一个 Vid/sub_id 有多个人或多种情绪，会在 manifest 中出现多行。

输出文件：
  /data/wzw/egolink_race/data/manifest/emotion_sort_manifest.csv
"""

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

DATA_ROOT = Path("/data/wzw/egolink_race/data")

ANNOTATION = DATA_ROOT / "annotation" / "data2.xlsx"
VIDEO_ROOT = DATA_ROOT / "E3" / "E3"
AUDIO_ROOT = DATA_ROOT / "audio"
SUBTITLE_ROOT = DATA_ROOT / "subtext"
OUT_PATH = DATA_ROOT / "manifest" / "emotion_sort_manifest.csv"

EMOTIONS = [
    "angry",
    "disgusted",
    "happy",
    "sad",
    "sarcastic",
    "scared",
    "shy",
    "surprised",
]
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


def cell_col_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_xlsx_first_sheet(path):
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                text = "".join(
                    node.text or ""
                    for node in item.findall(".//a:t", namespace)
                )
                shared_strings.append(text)

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//a:sheetData/a:row", namespace):
            values = {}
            max_index = -1
            for cell in row.findall("a:c", namespace):
                col_index = cell_col_index(cell.attrib["r"])
                max_index = max(max_index, col_index)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", namespace)

                if value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text

                values[col_index] = value

            rows.append([values.get(idx, "") for idx in range(max_index + 1)])

    header = rows[0]
    records = []
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, row)))

    return pd.DataFrame(records)


df = read_xlsx_first_sheet(ANNOTATION)

rows = []
missing_video = 0
missing_audio = 0
missing_subtitle = 0
unknown_emotion = 0

for _, row in df.iterrows():
    vid = str(row["Vid"])
    sub_id = str(row["sub_id"])
    emotion = str(row["emotion"]).strip().lower()

    if emotion not in EMOTION_TO_ID:
        unknown_emotion += 1
        continue

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
        "person": row["person"],
        "emotion": emotion,
        "label": EMOTION_TO_ID[emotion],
        "degree": row["degree"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "reason": row["reason"],
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "subtitle_path": str(subtitle_path),
        "split": row["set"],
    })

manifest = pd.DataFrame(rows)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUT_PATH, index=False)

print("manifest saved:", OUT_PATH)
print("usable samples:", len(manifest))
print("missing_video:", missing_video)
print("missing_audio:", missing_audio)
print("missing_subtitle:", missing_subtitle)
print("unknown_emotion:", unknown_emotion)
print("emotion_to_id:", EMOTION_TO_ID)
print(manifest["split"].value_counts())
print(manifest["emotion"].value_counts().sort_index())
