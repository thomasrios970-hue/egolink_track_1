import pandas as pd
from pathlib import Path

MANIFEST = Path("/data/wzw/egolink_race/data/manifest/manifest.csv")
HUBERT_DIR = Path("/data/wzw/egolink_race/feature/hubert_large")
OUT = Path("/data/wzw/egolink_race/data/manifest/hubert_manifest.csv")

label_map = {-3: 0, -2: 1, -1: 2, 1: 3, 2: 4, 3: 5}

df = pd.read_csv(MANIFEST)

rows = []
missing = 0

for _, row in df.iterrows():
    vid = str(row["Vid"])
    sub_id = str(row["sub_id"])
    feat_path = HUBERT_DIR / f"{vid}_{sub_id}.npy"

    if not feat_path.exists():
        missing += 1
        continue

    item = row.to_dict()
    item["hubert_path"] = str(feat_path)
    item["label_raw"] = int(row["label"])
    item["label_id"] = label_map[int(row["label"])]
    rows.append(item)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)

print("saved:", OUT)
print("usable:", len(out_df))
print("missing_hubert:", missing)
print(out_df["split"].value_counts())
print(out_df["label_raw"].value_counts().sort_index())
#生成一个只有音频文件有的清单,并返回可用样本数量和不可用样本数量，以及每个样本集要的标签种类是多少