"""Video retrieval extension for the multimedia retrieval backend."""

try:  # pragma: no cover - dependency guard for lightweight imports
    from .service import VideoRetrievalService, VideoRetrievalConfig
except Exception:  # pragma: no cover - allow datasets/utilities without full stack
    VideoRetrievalService = None
    VideoRetrievalConfig = None

__all__ = ["VideoRetrievalService", "VideoRetrievalConfig"]
