from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint support
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.video_cli_common import build_parser, configure_service


def main() -> None:
    parser = build_parser("Search video content from a text query.")
    parser.add_argument("--query", required=True, help="Natural language text query")
    parser.add_argument("--rerank-alpha", type=float, default=None)
    args = parser.parse_args()
    service = configure_service(args)
    service.load_or_build(
        sampling_strategy=args.sampling,
        sample_fps=args.fps,
        fixed_frames=args.fixed_frames,
        force_rebuild=args.force,
    )
    mode = args.mode if args.mode in {"video", "frame"} else "video"
    payload = service.search(
        query=args.query,
        top_k=args.top_k,
        mode=mode,
        sampling_strategy=args.sampling,
        rerank=args.rerank,
        return_segments=args.return_segments,
        rerank_alpha=args.rerank_alpha,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
