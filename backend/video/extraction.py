from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class VideoInfo:
    video_id: str
    file_name: str
    relative_path: str
    file_path: str
    fps: float
    frame_count: int
    duration_sec: float
    source_bucket: str | None = None


def _import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "OpenCV is required for video ingestion. Install `opencv-python`."
        ) from exc


def inspect_video(video_path: str | Path, data_root: str | Path | None = None) -> VideoInfo:
    cv2 = _import_cv2()
    path = Path(video_path)
    relative_path = path.name
    source_bucket = None
    if data_root is not None:
        try:
            relative = path.resolve().relative_to(Path(data_root).resolve())
            relative_path = relative.as_posix()
            source_bucket = relative.parts[0] if len(relative.parts) > 1 else None
        except Exception:
            relative_path = path.name
    else:
        relative_path = path.name
    video_id = Path(relative_path).with_suffix("").as_posix().replace("/", "__")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not decode video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(frame_count / fps) if fps > 0 else 0.0
    capture.release()
    return VideoInfo(
        video_id=video_id,
        file_name=path.name,
        relative_path=relative_path,
        file_path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        source_bucket=source_bucket,
    )


def sample_frame_indices(
    frame_count: int,
    fps: float,
    sampling_strategy: str = "uniform",
    sample_fps: float = 1.0,
    fixed_frames: int = 16,
) -> list[int]:
    if frame_count <= 0:
        return []

    strategy = (sampling_strategy or "uniform").lower()
    if strategy == "fixed":
        count = max(1, int(fixed_frames))
        if count >= frame_count:
            return list(range(frame_count))
        return sorted(set(int(x) for x in np.linspace(0, frame_count - 1, count)))

    if fps > 0 and sample_fps > 0:
        step = max(1, int(round(fps / sample_fps)))
        return list(range(0, frame_count, step))

    count = max(1, int(fixed_frames))
    if count >= frame_count:
        return list(range(frame_count))
    return sorted(set(int(x) for x in np.linspace(0, frame_count - 1, count)))


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sampling_strategy: str = "uniform",
    sample_fps: float = 1.0,
    fixed_frames: int = 16,
    save_frames: bool = True,
    data_root: str | Path | None = None,
) -> tuple[VideoInfo, list[dict[str, Any]]]:
    cv2 = _import_cv2()
    info = inspect_video(video_path, data_root=data_root)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not decode video: {video_path}")

    frame_indices = sample_frame_indices(
        info.frame_count,
        info.fps,
        sampling_strategy=sampling_strategy,
        sample_fps=sample_fps,
        fixed_frames=fixed_frames,
    )
    output_base = Path(output_dir) / info.video_id
    frame_store_dir = output_base if save_frames else output_base / "_tmp"
    frame_store_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        timestamp_sec = float(frame_index / info.fps) if info.fps > 0 else 0.0
        frame_path = None
        frame_path = frame_store_dir / f"frame_{frame_index:06d}_{int(timestamp_sec * 1000):010d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        records.append(
            {
                "video_id": info.video_id,
                "file_name": info.file_name,
                "relative_path": info.relative_path,
                "file_path": info.file_path,
                "frame_index": int(frame_index),
                "timestamp_sec": round(timestamp_sec, 3),
                "sampling_strategy": sampling_strategy,
                "extracted_frame_path": str(frame_path) if frame_path else None,
                "segment_id": None,
                "duration_sec": round(info.duration_sec, 3),
                "source_bucket": info.source_bucket,
            }
        )

    capture.release()
    return info, records
