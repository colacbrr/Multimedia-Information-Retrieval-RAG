import os


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEVICE = "cuda"
CLIP_MODEL_NAME = os.getenv("MIR_CLIP_MODEL", "ViT-B/32")
INDEX_TYPE = os.getenv("MIR_INDEX_TYPE", "flat").lower()
HNSW_M = int(os.getenv("MIR_HNSW_M", "32"))
HNSW_EF = int(os.getenv("MIR_HNSW_EF", "64"))
RERANK_ENABLED = env_bool("MIR_RERANK", True)
RERANK_ALPHA = float(os.getenv("MIR_RERANK_ALPHA", "0.25"))
RAG_MODEL = os.getenv("MIR_RAG_MODEL", "llama3.1:8b")
RAG_TEMPERATURE = float(os.getenv("MIR_RAG_TEMPERATURE", "0.2"))
RAG_TOP_P = float(os.getenv("MIR_RAG_TOP_P", "0.9"))
RAG_NUM_PREDICT = int(os.getenv("MIR_RAG_NUM_PREDICT", "256"))
RAG_TIMEOUT_SEC = float(os.getenv("MIR_RAG_TIMEOUT_SEC", "45"))
RAG_CACHE_TTL_SEC = float(os.getenv("MIR_RAG_CACHE_TTL_SEC", "300"))
RAG_MAX_CONTEXT_ITEMS = int(os.getenv("MIR_RAG_MAX_CONTEXT_ITEMS", "5"))
RAG_MAX_CAPTION_CHARS = int(os.getenv("MIR_RAG_MAX_CAPTION_CHARS", "180"))
MAX_IMAGES = int(os.getenv("MIR_MAX_IMAGES", "1000"))
PORT = int(os.getenv("MIR_PORT", "8000"))
SUPPORTED_MODALITIES = {"image", "video"}
FUTURE_MODALITIES = {"audio"}
PROMPT_VERSION = "v2"

VIDEO_DATA_DIR = os.getenv("VIDEO_DATA_DIR", "../data/videos")
VIDEO_ANNOTATIONS = os.getenv("VIDEO_ANNOTATIONS", "../data/videos/annotations/captions.json")
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "outputs/video")
VIDEO_SAMPLING_STRATEGY = os.getenv("VIDEO_SAMPLING_STRATEGY", "uniform")
VIDEO_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "1"))
VIDEO_FIXED_FRAMES = int(os.getenv("VIDEO_FIXED_FRAMES", "16"))
VIDEO_SAVE_FRAMES = env_bool("VIDEO_SAVE_FRAMES", True)
VIDEO_RERANK_ALPHA = float(os.getenv("VIDEO_RERANK_ALPHA", "0.7"))
VIDEO_MAX_RESULTS = int(os.getenv("VIDEO_MAX_RESULTS", "50"))
VIDEO_FRAME_BATCH_SIZE = int(os.getenv("VIDEO_FRAME_BATCH_SIZE", "16"))
VIDEO_CAPTION_BATCH_SIZE = int(os.getenv("VIDEO_CAPTION_BATCH_SIZE", "64"))
VIDEO_UPLOAD_BATCH_SIZE = int(os.getenv("VIDEO_UPLOAD_BATCH_SIZE", "8"))
