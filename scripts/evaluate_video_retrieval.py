from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint support
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from backend.video.datasets import load_local_captions
from scripts.video_cli_common import build_parser, configure_service


def main() -> None:
    parser = build_parser("Evaluate video retrieval using captions as queries.")
    parser.add_argument("--query-source", default=None, help="Optional captions JSON override")
    args = parser.parse_args()
    service = configure_service(args)
    service.load_or_build(
        sampling_strategy=args.sampling,
        sample_fps=args.fps,
        fixed_frames=args.fixed_frames,
        force_rebuild=args.force,
    )
    mode = args.mode if args.mode in {"video", "frame"} else "video"
    captions_path = Path(args.query_source or args.captions)
    captions = load_local_captions(captions_path) if captions_path.exists() else {}
    queries = []
    if isinstance(captions, dict):
        for video_id, record in captions.items():
            for caption in record.captions:
                if caption and video_id:
                    queries.append((caption, video_id))
    results = {"total": 0, "hits@1": 0, "hits@5": 0, "hits@10": 0}
    for caption, expected_video_id in queries:
        payload = service.search(
            query=caption,
            top_k=max(args.top_k, 10),
            mode=mode,
            sampling_strategy=args.sampling,
            rerank=args.rerank,
            return_segments=args.return_segments,
        )
        ranked = payload.get("results") or []
        ranked_ids = [item.get("video_id") for item in ranked]
        results["total"] += 1
        if expected_video_id in ranked_ids[:1]:
            results["hits@1"] += 1
        if expected_video_id in ranked_ids[:5]:
            results["hits@5"] += 1
        if expected_video_id in ranked_ids[:10]:
            results["hits@10"] += 1
    if results["total"] > 0:
        summary = {
            "total": results["total"],
            "recall@1": round(results["hits@1"] / results["total"], 4),
            "recall@5": round(results["hits@5"] / results["total"], 4),
            "recall@10": round(results["hits@10"] / results["total"], 4),
        }
    else:
        summary = {"total": 0, "recall@1": None, "recall@5": None, "recall@10": None}
    out = Path(service.config.output_dir) / "evaluation_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
