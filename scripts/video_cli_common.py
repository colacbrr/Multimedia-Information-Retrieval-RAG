from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import search_server as server


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--videos", default=str(server.VIDEO_DATA_PATH), help="Local video directory")
    parser.add_argument(
        "--captions",
        default=str(server.VIDEO_ANNOTATIONS_PATH),
        help="Local captions JSON file",
    )
    parser.add_argument("--output-dir", default=str(server.VIDEO_OUTPUTS_DIR), help="Video output directory")
    parser.add_argument("--sampling", default=server.video_service.config.sampling_strategy if server.video_service else "uniform")
    parser.add_argument("--fps", type=float, default=server.video_service.config.sample_fps if server.video_service else 1.0)
    parser.add_argument(
        "--fixed-frames",
        type=int,
        default=server.video_service.config.fixed_frames if server.video_service else 16,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["video", "frame", "both"], default="video")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true", help="Enable caption reranking")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable caption reranking")
    parser.set_defaults(rerank=True)
    segment_group = parser.add_mutually_exclusive_group()
    segment_group.add_argument("--return-segments", dest="return_segments", action="store_true", help="Return best timestamp and frame info")
    segment_group.add_argument("--no-return-segments", dest="return_segments", action="store_false", help="Skip best timestamp and frame info")
    parser.set_defaults(return_segments=True)
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists")
    return parser


def configure_service(args: argparse.Namespace):
    if server.video_service is None:
        raise RuntimeError("Video service is unavailable.")
    config = server.video_service.config
    config.data_dir = Path(args.videos)
    config.annotations_path = Path(args.captions)
    config.output_dir = Path(args.output_dir)
    config.frames_dir = config.output_dir / "frames"
    config.manifest_path = config.output_dir / "video_manifest.json"
    config.frame_embeddings_path = config.output_dir / "video_frame_embeddings.npy"
    config.frame_metadata_path = config.output_dir / "video_frame_metadata.json"
    config.video_embeddings_path = config.output_dir / "video_video_embeddings.npy"
    config.video_metadata_path = config.output_dir / "video_video_metadata.json"
    config.caption_embeddings_path = config.output_dir / "video_caption_embeddings.npy"
    config.caption_metadata_path = config.output_dir / "video_caption_metadata.json"
    config.frame_index_path = config.output_dir / "faiss_frames.index"
    config.video_index_path = config.output_dir / "faiss_video.index"
    config.caption_index_path = config.output_dir / "faiss_captions.index"
    config.metrics_path = config.output_dir / "video_metrics.jsonl"
    server.VIDEO_DATA_PATH = config.data_dir
    server.VIDEO_ANNOTATIONS_PATH = config.annotations_path
    server.VIDEO_OUTPUTS_DIR = config.output_dir
    server.VIDEO_FRAMES_DIR = config.frames_dir
    return server.video_service
