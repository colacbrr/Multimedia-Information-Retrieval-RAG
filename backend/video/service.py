from __future__ import annotations

import hashlib
import gc
import json
import os
import sys
import shutil
import time
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .datasets import ensure_directory, json_dump, list_video_files, load_local_captions
from .embeddings import EMBEDDING_DIM, aggregate_embeddings, encode_image_paths, encode_texts, l2_normalize
from .extraction import extract_frames, inspect_video
from .indexing import build_faiss_index, load_index, save_index, search_index
from .metrics import append_event, summarize_events
from .rag import build_fallback_video_summary, build_video_rag_prompt


@dataclass(slots=True)
class VideoRetrievalConfig:
    base_dir: Path
    data_dir: Path
    annotations_path: Path
    output_dir: Path
    frames_dir: Path
    manifest_path: Path
    frame_embeddings_path: Path
    frame_metadata_path: Path
    video_embeddings_path: Path
    video_metadata_path: Path
    caption_embeddings_path: Path
    caption_metadata_path: Path
    frame_index_path: Path
    video_index_path: Path
    caption_index_path: Path
    metrics_path: Path
    clip_model_name: str
    index_type: str
    sampling_strategy: str
    sample_fps: float
    fixed_frames: int
    save_frames: bool
    rerank_alpha: float
    max_results: int
    frame_batch_size: int
    caption_batch_size: int
    upload_batch_size: int


class VideoRetrievalService:
    def __init__(
        self,
        config: VideoRetrievalConfig,
        model_provider: Callable[[], tuple[Any, Any]],
        device_provider: Callable[[], str],
    ) -> None:
        self.config = config
        self.model_provider = model_provider
        self.device_provider = device_provider
        self.model = None
        self.preprocess = None
        self.ready = False
        self.error: str | None = None
        self.manifest: dict[str, Any] | None = None
        self.video_records: list[dict[str, Any]] = []
        self.frame_records: list[dict[str, Any]] = []
        self.caption_records: list[dict[str, Any]] = []
        self.video_record_by_id: dict[str, dict[str, Any]] = {}
        self.video_index_by_id: dict[str, int] = {}
        self.frame_indices_by_video_id: dict[str, list[int]] = {}
        self.caption_indices_by_video_id: dict[str, list[int]] = {}
        self.video_embeddings: np.ndarray | None = None
        self.frame_embeddings: np.ndarray | None = None
        self.caption_embeddings: np.ndarray | None = None
        self.video_index = None
        self.frame_index = None
        self.caption_index = None
        self._build_lock = threading.Lock()
        self.progress = {
            "stage": "idle",
            "message": "Idle",
            "current": 0,
            "total": 0,
            "percent": 0,
        }

    def _set_progress(self, stage: str, message: str, current: int = 0, total: int = 0) -> None:
        percent = int((current / total) * 100) if total else 0
        self.progress = {
            "stage": stage,
            "message": message,
            "current": current,
            "total": total,
            "percent": percent,
        }

    def _ensure_model(self) -> tuple[Any, Any, str]:
        if self.model is None or self.preprocess is None:
            self.model, self.preprocess = self.model_provider()
        return self.model, self.preprocess, self.device_provider()

    def release_resources(self) -> None:
        self.model = None
        self.preprocess = None
        self.manifest = None
        self.video_records = []
        self.frame_records = []
        self.caption_records = []
        self.video_record_by_id = {}
        self.video_index_by_id = {}
        self.frame_indices_by_video_id = {}
        self.caption_indices_by_video_id = {}
        self.video_embeddings = None
        self.frame_embeddings = None
        self.caption_embeddings = None
        self.video_index = None
        self.frame_index = None
        self.caption_index = None
        self.ready = False
        self.error = None
        self.progress = {
            "stage": "idle",
            "message": "Idle",
            "current": 0,
            "total": 0,
            "percent": 0,
        }
        gc.collect()
        try:
            import torch

            if self.device_provider() == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _video_signature(
        self,
        sampling_strategy: str,
        sample_fps: float,
        fixed_frames: int,
    ) -> str:
        videos = list_video_files(self.config.data_dir)
        payload = {
            "clip_model": self.config.clip_model_name,
            "index_type": self.config.index_type,
            "sampling_strategy": sampling_strategy,
            "sample_fps": sample_fps,
            "fixed_frames": fixed_frames,
            "videos": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime": int(path.stat().st_mtime),
                }
                for path in videos
            ],
            "captions": {
                "exists": self.config.annotations_path.exists(),
                "size": self.config.annotations_path.stat().st_size if self.config.annotations_path.exists() else 0,
                "mtime": int(self.config.annotations_path.stat().st_mtime) if self.config.annotations_path.exists() else 0,
            },
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _load_manifest(self) -> dict[str, Any] | None:
        if not self.config.manifest_path.exists():
            return None
        try:
            return json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        json_dump(self.config.manifest_path, manifest)

    def _log(self, message: str) -> None:
        print(f"[video] {message}", file=sys.stdout, flush=True)

    def _rebuild_lookup_tables(self) -> None:
        self.video_record_by_id = {
            record["video_id"]: record for record in self.video_records if record.get("video_id")
        }
        self.video_index_by_id = {
            record["video_id"]: index
            for index, record in enumerate(self.video_records)
            if record.get("video_id")
        }
        self.frame_indices_by_video_id = {}
        for index, record in enumerate(self.frame_records):
            video_id = record.get("video_id")
            if not video_id:
                continue
            self.frame_indices_by_video_id.setdefault(video_id, []).append(index)
        self.caption_indices_by_video_id = {}
        for index, record in enumerate(self.caption_records):
            video_id = record.get("video_id")
            if not video_id:
                continue
            self.caption_indices_by_video_id.setdefault(video_id, []).append(index)

    def _frame_cache_dir(self, video_id: str) -> Path:
        return self.config.frames_dir / video_id

    def _frame_cache_metadata_path(self, video_id: str) -> Path:
        return self._frame_cache_dir(video_id) / "_cache.json"

    def _frame_cache_signature(
        self,
        info: Any,
        sampling_strategy: str,
        sample_fps: float,
        fixed_frames: int,
    ) -> dict[str, Any]:
        file_path = Path(info.file_path)
        return {
            "video_id": info.video_id,
            "file_name": info.file_name,
            "relative_path": info.relative_path,
            "file_size": file_path.stat().st_size if file_path.exists() else None,
            "file_mtime": int(file_path.stat().st_mtime) if file_path.exists() else None,
            "fps": round(float(info.fps), 6),
            "frame_count": int(info.frame_count),
            "duration_sec": round(float(info.duration_sec), 3),
            "sampling_strategy": sampling_strategy,
            "sample_fps": float(sample_fps),
            "fixed_frames": int(fixed_frames),
        }

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _write_frame_cache_metadata(self, video_id: str, metadata: dict[str, Any]) -> None:
        json_dump(self._frame_cache_metadata_path(video_id), metadata)

    def _load_indexes(self) -> bool:
        required = [
            self.config.frame_embeddings_path,
            self.config.video_embeddings_path,
            self.config.frame_metadata_path,
            self.config.video_metadata_path,
            self.config.frame_index_path,
            self.config.video_index_path,
        ]
        if not all(path.exists() for path in required):
            return False
        self.manifest = self._load_manifest()
        if not self.manifest:
            return False
        self.frame_embeddings = np.load(self.config.frame_embeddings_path)
        self.video_embeddings = np.load(self.config.video_embeddings_path)
        self.frame_records = json.loads(self.config.frame_metadata_path.read_text(encoding="utf-8"))
        self.video_records = json.loads(self.config.video_metadata_path.read_text(encoding="utf-8"))
        self.frame_index = load_index(self.config.frame_index_path)
        self.video_index = load_index(self.config.video_index_path)
        if self.config.caption_embeddings_path.exists() and self.config.caption_metadata_path.exists() and self.config.caption_index_path.exists():
            self.caption_embeddings = np.load(self.config.caption_embeddings_path)
            self.caption_records = json.loads(self.config.caption_metadata_path.read_text(encoding="utf-8"))
            self.caption_index = load_index(self.config.caption_index_path)
        self._rebuild_lookup_tables()
        self.ready = True
        self.error = None
        return True

    def load_or_build(
        self,
        sampling_strategy: str | None = None,
        sample_fps: float | None = None,
        fixed_frames: int | None = None,
        force_rebuild: bool = False,
    ) -> None:
        sampling_strategy = (sampling_strategy or self.config.sampling_strategy).lower()
        sample_fps = float(sample_fps if sample_fps is not None else self.config.sample_fps)
        fixed_frames = int(fixed_frames if fixed_frames is not None else self.config.fixed_frames)
        signature = self._video_signature(sampling_strategy, sample_fps, fixed_frames)
        manifest = self._load_manifest()
        if not force_rebuild and manifest and manifest.get("signature") == signature:
            if self._load_indexes():
                return
        with self._build_lock:
            manifest = self._load_manifest()
            if not force_rebuild and manifest and manifest.get("signature") == signature:
                if self._load_indexes():
                    return
            self.build(
                sampling_strategy=sampling_strategy,
                sample_fps=sample_fps,
                fixed_frames=fixed_frames,
                force_rebuild=force_rebuild,
            )

    def load_existing(self) -> bool:
        loaded = self._load_indexes()
        if not loaded:
            self.ready = False
            self.error = "Video index not loaded. Build it explicitly with scripts/build_video_index.py."
        return loaded

    def build(
        self,
        sampling_strategy: str | None = None,
        sample_fps: float | None = None,
        fixed_frames: int | None = None,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        model, preprocess, device = self._ensure_model()
        sampling_strategy = (sampling_strategy or self.config.sampling_strategy).lower()
        sample_fps = float(sample_fps if sample_fps is not None else self.config.sample_fps)
        fixed_frames = int(fixed_frames if fixed_frames is not None else self.config.fixed_frames)

        ensure_directory(self.config.output_dir)
        ensure_directory(self.config.frames_dir)
        videos = list_video_files(self.config.data_dir)
        if not videos:
            raise FileNotFoundError(f"No video files found in {self.config.data_dir}")

        captions = load_local_captions(self.config.annotations_path)
        self._log(
            f"Building video index from {len(videos)} videos in {self.config.data_dir}"
        )
        self._set_progress("video-init", "Extracting frames", 0, len(videos))

        frame_records: list[dict[str, Any]] = []
        video_records: list[dict[str, Any]] = []
        for idx, video_path in enumerate(videos, start=1):
            info = inspect_video(video_path, data_root=self.config.data_dir)
            extracted: list[dict[str, Any]] = []
            existing_frame_dir = self._frame_cache_dir(info.video_id)
            expected_cache_meta = self._frame_cache_signature(
                info,
                sampling_strategy=sampling_strategy,
                sample_fps=sample_fps,
                fixed_frames=fixed_frames,
            )
            existing_meta = self._load_json(self._frame_cache_metadata_path(info.video_id))
            legacy_cache = (
                self.config.save_frames
                and existing_frame_dir.exists()
                and any(existing_frame_dir.glob("*.jpg"))
                and existing_meta is None
            )
            cache_matches = (
                self.config.save_frames
                and existing_frame_dir.exists()
                and bool(existing_meta)
                and existing_meta == expected_cache_meta
            )
            if self.config.save_frames and existing_frame_dir.exists() and not (cache_matches or legacy_cache):
                if force_rebuild or existing_meta is not None or any(existing_frame_dir.glob("*.jpg")):
                    shutil.rmtree(existing_frame_dir, ignore_errors=True)
            if cache_matches or legacy_cache:
                cached_paths = sorted(
                    path
                    for path in existing_frame_dir.glob("*.jpg")
                    if path.is_file()
                )
                if cached_paths:
                    for frame_path in cached_paths:
                        stem_parts = frame_path.stem.split("_")
                        try:
                            frame_index = int(stem_parts[1])
                        except Exception:
                            frame_index = 0
                        try:
                            timestamp_sec = int(stem_parts[2]) / 1000.0
                        except Exception:
                            timestamp_sec = 0.0
                        extracted.append(
                            {
                                "video_id": info.video_id,
                                "file_name": info.file_name,
                                "relative_path": info.relative_path,
                                "file_path": info.file_path,
                                "frame_index": frame_index,
                                "timestamp_sec": round(timestamp_sec, 3),
                                "sampling_strategy": sampling_strategy,
                                "extracted_frame_path": str(frame_path),
                                "segment_id": None,
                                "duration_sec": round(info.duration_sec, 3),
                                "source_bucket": info.source_bucket,
                            }
                        )
                    if idx == 1 or idx % 100 == 0 or idx == len(videos):
                        self._log(
                            f"Reused {len(extracted)} cached frames for {Path(video_path).name}"
                        )
                    if legacy_cache:
                        self._write_frame_cache_metadata(info.video_id, expected_cache_meta)
            if not extracted:
                info, extracted = extract_frames(
                    video_path,
                    self.config.frames_dir,
                    sampling_strategy=sampling_strategy,
                    sample_fps=sample_fps,
                    fixed_frames=fixed_frames,
                    save_frames=self.config.save_frames,
                    data_root=self.config.data_dir,
                )
                if self.config.save_frames:
                    self._write_frame_cache_metadata(info.video_id, expected_cache_meta)
            frame_start = len(frame_records)
            frame_records.extend(extracted)
            caption_record = captions.get(info.video_id)
            video_records.append(
                {
                    "video_id": info.video_id,
                    "file_name": info.file_name,
                    "relative_path": info.relative_path,
                    "file_path": info.file_path,
                    "duration_sec": round(info.duration_sec, 3),
                    "fps": round(info.fps, 3),
                    "frame_count": len(extracted),
                    "sampling_strategy": sampling_strategy,
                    "sample_fps": sample_fps,
                    "fixed_frames": fixed_frames,
                    "frame_start": frame_start,
                    "frame_end": len(frame_records),
                    "caption_count": len(caption_record.captions) if caption_record else 0,
                    "caption": caption_record.captions[0] if caption_record and caption_record.captions else None,
                    "source_bucket": info.source_bucket,
                }
            )
            if idx == 1 or idx % 100 == 0 or idx == len(videos):
                self._log(
                    f"Extracted frames for {idx}/{len(videos)} videos; "
                    f"collected {len(frame_records)} sampled frames"
                )
            self._set_progress("video-init", "Extracting frames", idx, len(videos))

        frame_paths = [record["extracted_frame_path"] for record in frame_records if record.get("extracted_frame_path")]
        self._set_progress("encode-frames", "Encoding extracted frames", 0, len(frame_paths))
        frame_embeddings, valid_paths = encode_image_paths(
            model,
            preprocess,
            device,
            frame_paths,
            batch_size=self.config.frame_batch_size,
        )
        self._set_progress("encode-frames", "Encoding extracted frames", len(valid_paths), len(frame_paths))
        self._log(
            f"Encoded {len(valid_paths)} frames into embeddings "
            f"with shape {list(frame_embeddings.shape)}"
        )
        valid_path_to_index = {path: idx for idx, path in enumerate(valid_paths)}
        self.frame_records = [record for record in frame_records if record.get("extracted_frame_path") in valid_paths]
        for record in self.frame_records:
            record["embedding_index"] = valid_path_to_index[record["extracted_frame_path"]]
        self.frame_embeddings = frame_embeddings
        np.save(self.config.frame_embeddings_path, self.frame_embeddings)
        json_dump(self.config.frame_metadata_path, self.frame_records)
        self._log(
            f"Checkpoint saved: frame embeddings -> {self.config.frame_embeddings_path.name}, "
            f"frame metadata -> {self.config.frame_metadata_path.name}"
        )

        video_embeddings: list[np.ndarray] = []
        final_video_records: list[dict[str, Any]] = []
        video_to_frame_indices: dict[str, list[int]] = {}
        for record in self.frame_records:
            video_to_frame_indices.setdefault(record["video_id"], []).append(record["embedding_index"])
        self._set_progress("aggregate-videos", "Aggregating video embeddings", 0, len(video_records))
        for video_record in video_records:
            indices = video_to_frame_indices.get(video_record["video_id"], [])
            if not indices:
                continue
            pooled = aggregate_embeddings(self.frame_embeddings[indices], method="mean")
            video_embeddings.append(pooled[0])
            video_record["frame_indices"] = indices
            final_video_records.append(video_record)
            self._set_progress("aggregate-videos", "Aggregating video embeddings", len(final_video_records), len(video_records))

        if not video_embeddings:
            raise ValueError("No valid frame embeddings were produced for videos.")

        self.video_embeddings = l2_normalize(np.vstack(video_embeddings))
        self.video_records = final_video_records
        self._rebuild_lookup_tables()
        np.save(self.config.video_embeddings_path, self.video_embeddings)
        json_dump(self.config.video_metadata_path, self.video_records)
        self._log(
            f"Checkpoint saved: video embeddings -> {self.config.video_embeddings_path.name}, "
            f"video metadata -> {self.config.video_metadata_path.name}"
        )

        caption_records: list[dict[str, Any]] = []
        caption_texts: list[str] = []
        for video_record in self.video_records:
            caption_record = captions.get(video_record["video_id"])
            if not caption_record:
                continue
            for caption_index, caption in enumerate(caption_record.captions):
                caption_records.append(
                    {
                        "video_id": video_record["video_id"],
                        "file_name": caption_record.file_name,
                        "relative_path": video_record.get("relative_path"),
                        "caption_index": caption_index,
                        "caption": caption,
                    }
                )
                caption_texts.append(caption)

        self.caption_records = caption_records
        if caption_texts:
            import clip

            self._set_progress("encode-captions", "Encoding caption texts", 0, len(caption_texts))
            self.caption_embeddings = encode_texts(
                model,
                device,
                caption_texts,
                clip.tokenize,
                batch_size=self.config.caption_batch_size,
                progress_cb=lambda current, total: self._set_progress(
                    "encode-captions", "Encoding caption texts", current, total
                ),
            )
            self._set_progress(
                "encode-captions",
                "Encoding caption texts",
                len(caption_texts),
                len(caption_texts),
            )
        else:
            self.caption_embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._rebuild_lookup_tables()
        np.save(self.config.caption_embeddings_path, self.caption_embeddings)
        json_dump(self.config.caption_metadata_path, self.caption_records)
        self._log(
            f"Checkpoint saved: caption embeddings -> {self.config.caption_embeddings_path.name}, "
            f"caption metadata -> {self.config.caption_metadata_path.name}"
        )

        self._set_progress("build-index", "Building FAISS frame index", 0, len(self.frame_records))
        self.frame_index = build_faiss_index(self.frame_embeddings, self.config.index_type)
        self._set_progress("build-index", "Building FAISS video index", 0, len(self.video_records))
        self.video_index = build_faiss_index(self.video_embeddings, self.config.index_type)
        self.caption_index = None
        if len(caption_texts):
            self._set_progress("build-index", "Building FAISS caption index", 0, len(self.caption_records))
            self.caption_index = build_faiss_index(self.caption_embeddings, self.config.index_type)

        save_index(self.frame_index, self.config.frame_index_path)
        save_index(self.video_index, self.config.video_index_path)
        if self.caption_index is not None:
            save_index(self.caption_index, self.config.caption_index_path)

        np.save(self.config.frame_embeddings_path, self.frame_embeddings)
        np.save(self.config.video_embeddings_path, self.video_embeddings)
        json_dump(self.config.frame_metadata_path, self.frame_records)
        json_dump(self.config.video_metadata_path, self.video_records)
        np.save(self.config.caption_embeddings_path, self.caption_embeddings)
        json_dump(self.config.caption_metadata_path, self.caption_records)

        signature = self._video_signature(sampling_strategy, sample_fps, fixed_frames)
        self.manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "signature": signature,
            "clip_model": self.config.clip_model_name,
            "index_type": self.config.index_type,
            "sampling_strategy": sampling_strategy,
            "sample_fps": sample_fps,
            "fixed_frames": fixed_frames,
            "video_count": len(self.video_records),
            "frame_count": len(self.frame_records),
            "caption_count": len(self.caption_records),
            "video_embeddings_shape": list(self.video_embeddings.shape),
            "frame_embeddings_shape": list(self.frame_embeddings.shape),
            "caption_embeddings_shape": list(self.caption_embeddings.shape),
            "video_embeddings_file": self.config.video_embeddings_path.name,
            "frame_embeddings_file": self.config.frame_embeddings_path.name,
        }
        self._save_manifest(self.manifest)
        self.ready = True
        self.error = None
        self._set_progress("ready", "Video index ready", len(self.video_records), len(self.video_records))
        return self.manifest

    def _best_frame_for_video(self, query_vector: np.ndarray, video_record: dict[str, Any]) -> dict[str, Any]:
        video_id = video_record.get("video_id")
        indices = self.frame_indices_by_video_id.get(video_id or "", video_record.get("frame_indices") or [])
        if not indices or self.frame_embeddings is None:
            return {"best_timestamp": None, "best_frame_path": None, "best_frame_score": None, "segment_id": None}
        vectors = self.frame_embeddings[indices]
        scores = (vectors @ query_vector).reshape(-1)
        best_local = int(np.argmax(scores))
        frame_record = self.frame_records[indices[best_local]]
        return {
            "best_timestamp": frame_record.get("timestamp_sec"),
            "best_frame_path": frame_record.get("extracted_frame_path"),
            "best_frame_score": float(scores[best_local]),
            "segment_id": frame_record.get("segment_id"),
            "frame_index": frame_record.get("frame_index"),
            "caption_context": frame_record.get("caption_context"),
        }

    def _caption_score(self, query_vector: np.ndarray, video_id: str) -> tuple[float | None, str | None]:
        if self.caption_embeddings is None or not len(self.caption_records):
            return None, None
        indices = self.caption_indices_by_video_id.get(video_id, [])
        if not indices:
            return None, None
        scores = self.caption_embeddings[indices] @ query_vector
        best_local = int(np.argmax(scores))
        return float(scores[best_local]), self.caption_records[indices[best_local]].get("caption")

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "video",
        sampling_strategy: str | None = None,
        rerank: bool = True,
        return_segments: bool = True,
        rerank_alpha: float | None = None,
    ) -> dict[str, Any]:
        if not self.ready:
            if not self.load_existing():
                raise RuntimeError(
                    "Video index is not loaded. Build it explicitly with scripts/build_video_index.py."
                )
        model, preprocess, device = self._ensure_model()
        import clip

        start = time.perf_counter()
        query_vector = encode_texts(model, device, [query], clip.tokenize)
        if query_vector.size == 0:
            raise ValueError("Empty query embedding.")
        query_vector = query_vector[0]

        mode = (mode or "video").lower()
        if mode not in {"video", "frame"}:
            mode = "video"
        rerank_alpha = float(rerank_alpha if rerank_alpha is not None else self.config.rerank_alpha)
        candidate_k = min(max(top_k * 4, top_k), len(self.video_records) if mode == "video" else len(self.frame_records))
        if mode == "frame" and self.frame_index is not None:
            scores, indices = search_index(self.frame_index, query_vector, top_k=candidate_k)
            grouped: dict[str, dict[str, Any]] = {}
            for frame_score, frame_idx in zip(scores, indices):
                if frame_idx < 0 or frame_idx >= len(self.frame_records):
                    continue
                frame_record = self.frame_records[frame_idx]
                video_id = frame_record["video_id"]
                bucket = grouped.setdefault(
                    video_id,
                    {
                        "video_id": video_id,
                        "video_file": frame_record.get("file_name"),
                        "file_name": frame_record.get("file_name"),
                        "relative_path": frame_record.get("relative_path"),
                        "source_bucket": frame_record.get("source_bucket"),
                        "visual_scores": [],
                        "best_frame_score": float(frame_score),
                        "best_timestamp": frame_record.get("timestamp_sec"),
                        "best_frame_path": frame_record.get("extracted_frame_path"),
                        "segment_id": frame_record.get("segment_id"),
                        "caption_context": frame_record.get("caption_context"),
                    },
                )
                bucket["visual_scores"].append(float(frame_score))
                if float(frame_score) > float(bucket["best_frame_score"]):
                    bucket["best_frame_score"] = float(frame_score)
                    bucket["best_timestamp"] = frame_record.get("timestamp_sec")
                    bucket["best_frame_path"] = frame_record.get("extracted_frame_path")
                    bucket["segment_id"] = frame_record.get("segment_id")
            candidates = list(grouped.values())
            for item in candidates:
                scores_list = sorted(item["visual_scores"], reverse=True)[:3]
                item["visual_score"] = float(sum(scores_list) / len(scores_list)) if scores_list else 0.0
                caption_score, caption_text = self._caption_score(query_vector, item["video_id"])
                item["caption_score"] = caption_score
                if caption_text and not item.get("caption_context"):
                    item["caption_context"] = caption_text
                if rerank and caption_score is not None:
                    item["score"] = float(rerank_alpha * item["visual_score"] + (1.0 - rerank_alpha) * caption_score)
                else:
                    item["score"] = float(item["visual_score"])
                item["retrieval_mode"] = "frame"
                item["video_url"] = f"/video-files/{item.get('relative_path') or item['file_name']}"
                if item.get("best_frame_path"):
                    try:
                        rel_frame = Path(item["best_frame_path"]).resolve().relative_to(self.config.frames_dir)
                        item["best_frame_url"] = f"/video-frames/{rel_frame.as_posix()}"
                    except Exception:
                        item["best_frame_url"] = None
            results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
        else:
            if self.video_index is None:
                raise RuntimeError("Video index is not available.")
            scores, indices = search_index(self.video_index, query_vector, top_k=candidate_k)
            results = []
            for video_score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.video_records):
                    continue
                video_record = self.video_records[idx]
                frame_info = self._best_frame_for_video(query_vector, video_record) if return_segments else {}
                caption_score, caption_text = self._caption_score(query_vector, video_record["video_id"])
                score = float(video_score)
                if rerank and caption_score is not None:
                    score = float(rerank_alpha * float(video_score) + (1.0 - rerank_alpha) * caption_score)
                result = {
                    "video_id": video_record["video_id"],
                    "video_file": video_record["file_name"],
                    "file_name": video_record["file_name"],
                    "relative_path": video_record.get("relative_path"),
                    "source_bucket": video_record.get("source_bucket"),
                    "video_url": f"/video-files/{video_record.get('relative_path') or video_record['file_name']}",
                    "score": score,
                    "visual_score": float(video_score),
                    "caption_score": caption_score,
                    "caption": caption_text,
                    "retrieval_mode": "video",
                    "best_timestamp": frame_info.get("best_timestamp"),
                    "best_frame_path": frame_info.get("best_frame_path"),
                    "best_frame_score": frame_info.get("best_frame_score"),
                    "segment_id": frame_info.get("segment_id"),
                    "caption_context": frame_info.get("caption_context"),
                    "duration_sec": video_record.get("duration_sec"),
                    "frame_count": video_record.get("frame_count"),
                }
                if result["best_frame_path"]:
                    try:
                        rel_frame = Path(result["best_frame_path"]).resolve().relative_to(self.config.frames_dir)
                        result["best_frame_url"] = f"/video-frames/{rel_frame.as_posix()}"
                    except Exception:
                        result["best_frame_url"] = None
                results.append(result)
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "rerank": rerank,
            "rerank_alpha": rerank_alpha,
            "sampling_strategy": sampling_strategy or self.config.sampling_strategy,
            "return_segments": return_segments,
            "results": results,
        }
        append_event(
            self.config.metrics_path,
            {
                "type": "search",
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
                "indexed_videos": len(self.video_records),
                "indexed_frames": len(self.frame_records),
                "retrieval_backend": "faiss",
                "rerank": rerank,
            },
        )
        return payload

    def reverse_search(
        self,
        video_id: str | None = None,
        video_path: str | None = None,
        top_k: int = 5,
        mode: str = "video",
    ) -> dict[str, Any]:
        if not self.ready:
            if not self.load_existing():
                raise RuntimeError(
                    "Video index is not loaded. Build it explicitly with scripts/build_video_index.py."
                )
        model, preprocess, device = self._ensure_model()
        import clip
        start = time.perf_counter()

        if video_path:
            with tempfile.TemporaryDirectory(prefix="reverse_", dir=str(self.config.output_dir)) as tmp_dir:
                info, records = extract_frames(
                    video_path,
                    tmp_dir,
                    sampling_strategy=self.config.sampling_strategy,
                    sample_fps=self.config.sample_fps,
                    fixed_frames=self.config.fixed_frames,
                    save_frames=True,
                    data_root=self.config.data_dir,
                )
                if not records:
                    raise ValueError("No frames extracted from provided video.")
                frame_paths = [
                    record["extracted_frame_path"]
                    for record in records
                    if record.get("extracted_frame_path")
                ]
                vectors, _ = encode_image_paths(
                    model,
                    preprocess,
                    device,
                    frame_paths,
                    batch_size=self.config.upload_batch_size,
                )
                if vectors.size == 0:
                    raise ValueError("No valid frames could be encoded from provided video.")
                query_vector = aggregate_embeddings(vectors, method="mean")[0]
            video_label = info.relative_path
        else:
            if not video_id:
                raise ValueError("Either video_id or video_path is required.")
            match = self.video_record_by_id.get(video_id)
            if not match:
                raise KeyError(f"Unknown video_id: {video_id}")
            video_index = self.video_index_by_id.get(video_id)
            if video_index is None or self.video_embeddings is None:
                raise KeyError(f"Unknown video_id: {video_id}")
            query_vector = self.video_embeddings[video_index]
            video_label = match.get("relative_path") or match["file_name"]

        mode = (mode or "video").lower()
        if mode not in {"video", "frame"}:
            mode = "video"
        if self.caption_index is None or not len(self.caption_records):
            frame_records = (
                [record for record in self.frame_records if record["video_id"] == video_id][:top_k]
                if video_id
                else self.frame_records[:top_k]
            )
            fallback = [
                {
                    "video_id": record.get("video_id"),
                    "video_file": record.get("file_name"),
                    "video_url": f"/video-files/{record.get('relative_path') or record.get('file_name')}",
                    "relative_path": record.get("relative_path"),
                    "source_bucket": record.get("source_bucket"),
                    "caption": f"Frame {record.get('frame_index')} la {record.get('timestamp_sec')}s",
                    "score": 0.0,
                    "timestamp": record.get("timestamp_sec"),
                }
                for record in frame_records
            ]
            payload = {"video": video_label, "mode": mode, "results": fallback, "fallback": True}
            payload["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
            append_event(
                self.config.metrics_path,
                {
                    "type": "reverse_search",
                    "query": video_label,
                    "mode": mode,
                    "top_k": top_k,
                    "latency_ms": payload["latency_ms"],
                    "indexed_videos": len(self.video_records),
                    "indexed_frames": len(self.frame_records),
                    "retrieval_backend": "faiss",
                    "rerank": True,
                },
            )
            return payload

        scores, indices = search_index(self.caption_index, query_vector, top_k=top_k)
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.caption_records):
                continue
            record = self.caption_records[idx]
            results.append(
                {
                    "video_id": record["video_id"],
                    "video_file": record["file_name"],
                    "video_url": f"/video-files/{record.get('relative_path') or record.get('file_name')}",
                    "relative_path": record.get("relative_path"),
                    "source_bucket": record.get("source_bucket"),
                    "caption": record["caption"],
                    "caption_index": record["caption_index"],
                    "score": float(score),
                    "timestamp": None,
                    "mode": mode,
                }
            )
        payload = {"video": video_label, "mode": mode, "results": results, "fallback": False}
        payload["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        append_event(
            self.config.metrics_path,
            {
                "type": "reverse_search",
                "query": video_label,
                "mode": mode,
                "top_k": top_k,
                "latency_ms": payload["latency_ms"],
                "indexed_videos": len(self.video_records),
                "indexed_frames": len(self.frame_records),
                "retrieval_backend": "faiss",
                "rerank": True,
            },
        )
        return payload

    def rag(
        self,
        query: str,
        results: list[dict[str, Any]],
        model_name: str | None = None,
    ) -> dict[str, Any]:
        selected_model = model_name or os.getenv("MIR_VIDEO_RAG_MODEL", os.getenv("MIR_RAG_MODEL", "llama3.1:8b"))
        start = time.perf_counter()
        try:
            import ollama  # type: ignore
        except Exception:
            answer = build_fallback_video_summary(query, results, reason="Ollama client not available")
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            append_event(
                self.config.metrics_path,
                {
                    "type": "rag",
                    "query": query,
                    "model": selected_model,
                    "latency_ms": latency_ms,
                    "indexed_videos": len(self.video_records),
                    "indexed_frames": len(self.frame_records),
                    "retrieval_backend": "faiss",
                    "rerank": True,
                    "fallback": True,
                    "fallback_reason": "ollama_missing",
                },
            )
            return {
                "query": query,
                "model": selected_model,
                "latency_ms": latency_ms,
                "answer": answer,
                "prompt": build_video_rag_prompt(query, results),
                "fallback": True,
                "reason": "Ollama client not available",
            }

        prompt = build_video_rag_prompt(query, results)
        try:
            response = ollama.generate(
                model=selected_model,
                prompt=prompt,
                options={
                    "temperature": float(os.getenv("VIDEO_RAG_TEMPERATURE", os.getenv("MIR_RAG_TEMPERATURE", "0.2"))),
                    "top_p": float(os.getenv("VIDEO_RAG_TOP_P", os.getenv("MIR_RAG_TOP_P", "0.9"))),
                    "num_predict": int(os.getenv("VIDEO_RAG_NUM_PREDICT", os.getenv("MIR_RAG_NUM_PREDICT", "256"))),
                },
            )
        except Exception as exc:
            answer = build_fallback_video_summary(query, results, reason=f"Ollama error: {exc}")
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            append_event(
                self.config.metrics_path,
                {
                    "type": "rag",
                    "query": query,
                    "model": selected_model,
                    "latency_ms": latency_ms,
                    "indexed_videos": len(self.video_records),
                    "indexed_frames": len(self.frame_records),
                    "retrieval_backend": "faiss",
                    "rerank": True,
                    "fallback": True,
                    "fallback_reason": "ollama_error",
                },
            )
            return {
                "query": query,
                "model": selected_model,
                "latency_ms": latency_ms,
                "answer": answer,
                "prompt": prompt,
                "fallback": True,
                "reason": f"Ollama error: {exc}",
            }
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        answer = response.get("response", "") or response.get("message", "")
        append_event(
            self.config.metrics_path,
            {
                "type": "rag",
                "query": query,
                "model": selected_model,
                "latency_ms": latency_ms,
                "indexed_videos": len(self.video_records),
                "indexed_frames": len(self.frame_records),
                "retrieval_backend": "faiss",
                "rerank": True,
            },
        )
        return {
            "query": query,
            "model": selected_model,
            "latency_ms": latency_ms,
            "answer": answer,
            "prompt": prompt,
            "fallback": False,
        }

    def metrics_summary(self, limit: int = 1000) -> dict[str, Any]:
        summary = summarize_events(self.config.metrics_path, limit=limit)
        summary["indexed_videos"] = len(self.video_records)
        summary["indexed_frames"] = len(self.frame_records)
        return summary

    def status(self) -> dict[str, Any]:
        try:
            device = self.device_provider()
        except Exception:
            device = "unknown"
        return {
            "ready": self.ready,
            "error": self.error,
            "progress": self.progress,
            "manifest": self.manifest,
            "device": device,
            "data_dir": str(self.config.data_dir),
            "annotations_path": str(self.config.annotations_path),
            "indexed_videos": len(self.video_records),
            "indexed_frames": len(self.frame_records),
            "indexed_captions": len(self.caption_records),
            "sampling_strategy": self.manifest.get("sampling_strategy") if self.manifest else self.config.sampling_strategy,
            "sample_fps": self.manifest.get("sample_fps") if self.manifest else self.config.sample_fps,
            "fixed_frames": self.manifest.get("fixed_frames") if self.manifest else self.config.fixed_frames,
        }
