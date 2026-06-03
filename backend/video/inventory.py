from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
METADATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".bib"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".rar"}


@dataclass(slots=True)
class InventoryItem:
    name: str
    path: str
    exists: bool
    role: str
    video_files: int
    metadata_files: int
    archive_files: int
    total_files: int
    total_bytes: int
    ready_for_indexing: bool
    sample_files: list[str]


def _human_bytes(value: int) -> str:
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _summarize_directory(path: Path, name: str, role: str) -> InventoryItem:
    if not path.exists():
        return InventoryItem(
            name=name,
            path=str(path),
            exists=False,
            role=role,
            video_files=0,
            metadata_files=0,
            archive_files=0,
            total_files=0,
            total_bytes=0,
            ready_for_indexing=False,
            sample_files=[],
        )

    video_files = 0
    metadata_files = 0
    archive_files = 0
    total_files = 0
    total_bytes = 0
    sample_files: list[str] = []
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        total_files += 1
        suffix = file_path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            video_files += 1
        elif suffix in METADATA_EXTENSIONS:
            metadata_files += 1
        elif suffix in ARCHIVE_EXTENSIONS:
            archive_files += 1
        if len(sample_files) < 6:
            sample_files.append(str(file_path.relative_to(path)))
        try:
            total_bytes += file_path.stat().st_size
        except OSError:
            continue

    return InventoryItem(
        name=name,
        path=str(path),
        exists=True,
        role=role,
        video_files=video_files,
        metadata_files=metadata_files,
        archive_files=archive_files,
        total_files=total_files,
        total_bytes=total_bytes,
        ready_for_indexing=video_files > 0,
        sample_files=sample_files,
    )


def collect_video_inventory(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    data_root = root / "data"
    videos_root = data_root / "videos"
    items = [
        _summarize_directory(videos_root, "Local video corpus", "index-source"),
        _summarize_directory(videos_root / "annotations", "Local captions", "metadata"),
        _summarize_directory(data_root / "video_benchmarks", "Normalized video benchmarks", "benchmark-normalized"),
        _summarize_directory(data_root / "external_video_metadata", "External video metadata", "reference"),
    ]

    sources = []
    for item in items:
        sources.append(
            {
                "name": item.name,
                "path": item.path,
                "exists": item.exists,
                "role": item.role,
                "video_files": item.video_files,
                "metadata_files": item.metadata_files,
                "archive_files": item.archive_files,
                "total_files": item.total_files,
                "total_bytes": item.total_bytes,
                "size_human": _human_bytes(item.total_bytes),
                "ready_for_indexing": item.ready_for_indexing,
                "sample_files": item.sample_files,
            }
        )

    return {
        "project_root": str(root),
        "sources_root": str(data_root),
        "sources": sources,
        "ready_sources": sum(1 for item in sources if item.get("ready_for_indexing")),
        "video_files": sum(int(item.get("video_files", 0)) for item in sources),
        "metadata_files": sum(int(item.get("metadata_files", 0)) for item in sources),
        "archive_files": sum(int(item.get("archive_files", 0)) for item in sources),
        "total_files": sum(int(item.get("total_files", 0)) for item in sources),
        "total_bytes": sum(int(item.get("total_bytes", 0)) for item in sources),
        "size_human": _human_bytes(sum(int(item.get("total_bytes", 0)) for item in sources)),
    }
