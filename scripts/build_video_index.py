from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint support
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.video_cli_common import build_parser, configure_service


def main() -> None:
    parser = build_parser("Build the local video retrieval index.")
    args = parser.parse_args()
    service = configure_service(args)
    manifest = service.build(
        sampling_strategy=args.sampling,
        sample_fps=args.fps,
        fixed_frames=args.fixed_frames,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
