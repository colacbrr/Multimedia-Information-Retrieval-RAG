from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    payload = dict(event)
    payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_events(path: str | Path, limit: int = 1000) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "events": 0,
            "avg_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "indexed_videos": 0,
            "indexed_frames": 0,
        }

    events: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
        if limit and len(events) > limit:
            events = events[-limit:]
    except Exception:
        return {"events": 0}

    latencies = [
        float(event.get("latency_ms"))
        for event in events
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    indexed_videos = max(
        [int(event.get("indexed_videos", 0)) for event in events if isinstance(event.get("indexed_videos"), (int, float))]
        or [0]
    )
    indexed_frames = max(
        [int(event.get("indexed_frames", 0)) for event in events if isinstance(event.get("indexed_frames"), (int, float))]
        or [0]
    )
    return {
        "events": len(events),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "min_latency_ms": round(min(latencies), 2) if latencies else None,
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
        "indexed_videos": indexed_videos,
        "indexed_frames": indexed_frames,
    }
