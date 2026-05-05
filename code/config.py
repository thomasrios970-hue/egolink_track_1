from pathlib import Path


"""项目根路径"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


"""输入路径"""
PATH_TO_HUGGINGFACE_MODEL = resolve_path("model")
PATH_TO_DATA_DIR = resolve_path("data")
PATH_TO_ANNOTATION_DIR = PATH_TO_DATA_DIR / "annotation"
PATH_TO_VIDEO_ROOT = PATH_TO_DATA_DIR / "E3" / "E3"
PATH_TO_AUDIO_ROOT = PATH_TO_DATA_DIR / "audio"
PATH_TO_MANIFEST = PATH_TO_DATA_DIR / "manifest"

PATH_TO_DATA1_ANNOTATION = PATH_TO_ANNOTATION_DIR / "data1.xlsx"
PATH_TO_DATA2_ANNOTATION = PATH_TO_ANNOTATION_DIR / "data2.xlsx"

PATH_TO_AUDIO_MANIFEST = PATH_TO_MANIFEST / "emotion_degree_audio_manifest.csv"
PATH_TO_VIDEO_MANIFEST = PATH_TO_AUDIO_MANIFEST
PATH_TO_EMOTION_SORT_MANIFEST = PATH_TO_MANIFEST / "emotion_sort_manifest.csv"
PATH_TO_HUBERT_MANIFEST = PATH_TO_MANIFEST / "emotion_degree_hubert_manifest.csv"

"""输出路径"""
PATH_TO_PROCESSED_DIR = PATH_TO_DATA_DIR / "processed"
PATH_TO_FEATURE_DIR = resolve_path("feature")
PATH_TO_AUDIO_FEATURE_DIR = PATH_TO_FEATURE_DIR / "audio_features"
PATH_TO_TXT_FEATURE_DIR = PATH_TO_FEATURE_DIR / "txt_features"
PATH_TO_TARGET_TEXT_FEATURE_DIR = PATH_TO_FEATURE_DIR / "target_txt_features"
PATH_TO_VISUAL_FEATURE_DIR = PATH_TO_FEATURE_DIR / "visual_features"
PATH_TO_CHECKPOINT_DIR = resolve_path("checkpoints")
PATH_TO_LOG_DIR = resolve_path("log")
