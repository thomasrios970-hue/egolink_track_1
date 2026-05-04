"""
Extract row-aware target-person text features for emotion_sort.

Each manifest row gets its own feature file, keyed by row_id. This avoids the
Vid/sub_id collision that happens when one clip has labels for multiple people.

Text input is intentionally clean:
Target person: {person}. Subtitle: {subtitle_or_whisper_text}.

It does not read reason, tone_hint, emotion_annotations, or label-derived fields.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from download_model.download_models import download_models, txt_models


FEATURE_LEVELS = ["UTTERANCE"]
DEFAULT_MANIFEST = Path(config.PATH_TO_AUDIO_MANIFEST).parent / "emotion_sort_manifest.csv"
DEFAULT_SAVE_ROOT = Path(config.PATH_TO_TXT_FEATURE_DIR).parent / "target_txt_features"
DEFAULT_WHISPER_ROOT = Path(config.PATH_TO_PROCESSED_DIR) / "whisper" / "small"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract target-aware row_id text features for emotion_sort.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-name", default="xlm-roberta-xl")
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument(
        "--text-source",
        choices=["auto", "subtitle", "whisper"],
        default="auto",
        help="auto prefers whisper text when available, then falls back to subtitle_path.",
    )
    parser.add_argument("--whisper-root", type=Path, default=DEFAULT_WHISPER_ROOT)
    parser.add_argument("--feature-level", choices=FEATURE_LEVELS, default="UTTERANCE")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--max-length", type=positive_int, default=256)
    parser.add_argument("--progress-interval", type=positive_int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def get_row_id(row, row_index):
    if "row_id" in row and not pd.isna(row["row_id"]):
        text = str(row["row_id"]).strip()
        if text:
            return text
    return f"row_{int(row_index):06d}"


def read_vtt(path):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "WEBVTT":
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        lines.append(line)
    return " ".join(lines)


def get_whisper_path(row, whisper_root):
    return whisper_root / str(row["Vid"]) / f"{row['sub_id']}.vtt"


def get_clean_text(row, text_source, whisper_root):
    subtitle_text = ""
    whisper_text = ""

    if text_source in {"auto", "whisper"}:
        whisper_text = read_vtt(get_whisper_path(row, whisper_root))
    if text_source in {"auto", "subtitle"}:
        subtitle_path = Path(str(row.get("subtitle_path", "")))
        subtitle_text = read_vtt(subtitle_path)

    if text_source == "whisper":
        return whisper_text
    if text_source == "subtitle":
        return subtitle_text
    return whisper_text or subtitle_text


def build_target_text(row, text_source, whisper_root):
    person = str(row.get("person", "")).strip() or "the target person"
    subtitle = get_clean_text(row, text_source, whisper_root)
    subtitle = " ".join(subtitle.split()) or "[PAD]"
    return f"Target person: {person}. Subtitle: {subtitle}."


def ensure_model(model_name):
    model_path = Path(config.PATH_TO_HUGGINGFACE_MODEL) / model_name
    if (model_path / "config.json").exists():
        return model_path

    models_to_download = {
        repo_id: local_name
        for repo_id, local_name in txt_models.items()
        if local_name == model_name
    }
    if not models_to_download:
        raise ValueError(
            f"Local model is missing and {model_name} is not listed in txt_models: {model_path}"
        )
    download_models(models_to_download)
    return model_path


def extract_feature(text, tokenizer, model, device, max_length):
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        layers = torch.stack(hidden_states)[[-1, -2, -3, -4]].sum(dim=0)
        feature = layers.squeeze(0)
        utterance = feature[0].detach().cpu().numpy().astype(np.float32)
    return utterance


def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.manifest)
    required = {"Vid", "sub_id", "person"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    model_path = ensure_model(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = None
    if args.gpu != -1:
        device = torch.device(f"cuda:{args.gpu}")
        model.to(device)
    model.eval()

    out_dir = args.save_root / args.model_name / args.feature_level
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    for row_index, row in df.iterrows():
        row_id = get_row_id(row, row_index)
        save_path = out_dir / f"{row_id}.npy"
        if save_path.exists() and not args.overwrite:
            skipped += 1
            continue

        text = build_target_text(row, args.text_source, args.whisper_root)
        feature = extract_feature(text, tokenizer, model, device, args.max_length)
        np.save(save_path, feature)
        written += 1

        if (row_index + 1) % args.progress_interval == 0:
            print(
                f"[{row_index + 1}/{len(df)}] "
                f"written={written}, skipped={skipped}, last={save_path}"
            )

    print("Done")
    print(f"manifest={args.manifest}")
    print(f"save_dir={out_dir}")
    print(f"rows={len(df)}")
    print(f"written={written}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
