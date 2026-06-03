from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(slots=True)
class VideoCaptionRecord:
    video_id: str
    file_name: str
    captions: list[str]


def list_video_files(video_dir: str | Path) -> list[Path]:
    base = Path(video_dir)
    if not base.exists():
        return []
    discovered: dict[str, Path] = {}
    for ext in sorted(VIDEO_EXTENSIONS):
        for path in base.rglob(f"*{ext}"):
            if path.is_file():
                discovered[str(path.resolve())] = path
    return sorted(discovered.values())


def load_local_captions(captions_path: str | Path | None) -> dict[str, VideoCaptionRecord]:
    if not captions_path:
        return {}
    path = Path(captions_path)
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    records: dict[str, VideoCaptionRecord] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            video_id = str(item.get("video_id") or "").strip()
            file_name = str(item.get("file_name") or "").strip()
            captions = item.get("captions")
            if captions is None:
                # Intentionally only fall back when the key is missing or None.
                # An empty list means "no captions" and should stay empty.
                captions = item.get("caption")
            if not video_id and file_name:
                video_id = Path(file_name).stem
            if not video_id:
                continue
            if not isinstance(captions, list):
                captions = [str(captions)] if captions else []
            captions = [str(c).strip() for c in captions if str(c).strip()]
            if not file_name:
                file_name = str(item.get("video") or f"{video_id}.mp4").strip()
            records[video_id] = VideoCaptionRecord(
                video_id=video_id,
                file_name=file_name or f"{video_id}.mp4",
                captions=captions,
            )
    elif isinstance(raw, dict):
        for video_id, value in raw.items():
            captions: list[str] = []
            file_name = f"{video_id}.mp4"
            if isinstance(value, dict):
                file_name = str(value.get("file_name") or file_name)
                raw_captions = value.get("captions") or value.get("caption") or []
                if isinstance(raw_captions, list):
                    captions = [str(c).strip() for c in raw_captions if str(c).strip()]
                elif raw_captions:
                    captions = [str(raw_captions).strip()]
            elif isinstance(value, list):
                captions = [str(c).strip() for c in value if str(c).strip()]
            elif value:
                captions = [str(value).strip()]
            records[str(video_id)] = VideoCaptionRecord(
                video_id=str(video_id),
                file_name=file_name,
                captions=captions,
            )
    return records


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def json_dump(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
