from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint support
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.video_cli_common import build_parser, configure_service


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _prompt_int(prompt: str, default: int) -> int:
    raw = _prompt_text(prompt, str(default))
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    choices_set = {choice.lower() for choice in choices}
    raw = _prompt_text(prompt, default).lower()
    return raw if raw in choices_set else default


def _print_results(title: str, results: list[dict[str, object]]) -> None:
    print()
    print(title)
    print("=" * len(title))
    if not results:
        print("No results.")
        return
    for idx, item in enumerate(results, start=1):
        file_name = item.get("file_name") or item.get("video_file") or "unknown"
        score = item.get("score")
        timestamp = item.get("best_timestamp")
        caption = item.get("caption") or item.get("caption_context") or item.get("context") or "n/a"
        source = item.get("source_bucket") or "n/a"
        print(f"{idx}. {file_name}")
        print(f"   score: {score:.4f}" if isinstance(score, (int, float)) else f"   score: {score}")
        if isinstance(timestamp, (int, float)):
            print(f"   timestamp: {timestamp:.2f}s")
        else:
            print("   timestamp: necunoscut")
        print(f"   source: {source}")
        path = item.get("relative_path") or item.get("video_path") or item.get("path")
        video_url = item.get("video_url")
        if path:
            print(f"   path: {path}")
        if video_url:
            print(f"   url: {video_url}")
        print(f"   caption/context: {caption}")
        print()


def _print_payload_summary(payload: dict[str, object]) -> None:
    latency_ms = payload.get("latency_ms")
    results = payload.get("results") or []
    print(f"Latency: {latency_ms} ms")
    print(f"Results: {len(results)}")


def _default_demo_clip_path() -> Path | None:
    demo_dir = Path(__file__).resolve().parents[1] / "demo_inputs"
    if not demo_dir.exists():
        return None
    candidates = sorted(
        path
        for path in demo_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = build_parser("Interactive terminal demo for video retrieval and video-to-text summary.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama model for summary generation.",
    )
    parser.add_argument("--rerank-alpha", type=float, default=None)
    args = parser.parse_args()

    import search_server as server
    service = None

    try:
        print("Choose execution device:")
        print("  1) CPU")
        print("  2) GPU")
        print("  3) Auto")
        device_choice = _prompt_text("Selection", "1")
        if device_choice == "2":
            chosen_device = "cuda"
        elif device_choice == "3":
            chosen_device = "cuda" if server.torch.cuda.is_available() else "cpu"
        else:
            chosen_device = "cpu"

        server.DEVICE = chosen_device
        print(f"Using device: {server.DEVICE}")

        service = configure_service(args)
        if not service.load_existing():
            raise RuntimeError(
                "Video index is not loaded. Build it first with scripts/build_video_index.py."
            )

        print("Interactive video demo")
        print("----------------------")
        print(f"Index dir: {service.config.output_dir}")
        print(f"Data dir:  {service.config.data_dir}")
        print(f"Device:    {service.device_provider()}")

        while True:
            print()
            print("Choose mode:")
            print("  1) Text -> video retrieval")
            print("  2) Local clip -> video-to-text retrieval + summary")
            print("  3) Exit")
            choice = _prompt_text("Selection", "1")

            if choice == "3":
                print("Bye.")
                break

            if choice == "1":
                query = _prompt_text("Query")
                if not query:
                    print("Empty query, skipping.")
                    continue
                top_k = _prompt_int("Top-K", args.top_k)
                mode = _prompt_choice("Mode (video/frame)", ("video", "frame"), "video")
                rerank = _prompt_choice("Rerank? (y/n)", ("y", "n"), "y") == "y"
                return_segments = _prompt_choice("Return segments? (y/n)", ("y", "n"), "y") == "y"
                payload = service.search(
                    query=query,
                    top_k=top_k,
                    mode=mode,
                    sampling_strategy=args.sampling,
                    rerank=rerank,
                    return_segments=return_segments,
                    rerank_alpha=args.rerank_alpha,
                )
                _print_payload_summary(payload)
                _print_results("Top results", payload.get("results") or [])

                if payload.get("results"):
                    summarize = _prompt_choice("Generate summary explanation? (y/n)", ("y", "n"), "n") == "y"
                    if summarize:
                        rag = service.rag(
                            query=query,
                            results=(payload.get("results") or [])[:5],
                            model_name=args.model,
                        )
                        print()
                        print("Text-to-video summary")
                        print("=====================")
                        print(rag.get("answer") or rag.get("explanation") or rag.get("summary") or rag)
                service.release_resources()
                continue

            if choice == "2":
                demo_clip = _default_demo_clip_path()
                if demo_clip is None:
                    video_path = _prompt_text("Path to local video")
                    if not video_path:
                        print("Empty path, skipping.")
                        continue
                    path = Path(video_path).expanduser()
                else:
                    path = demo_clip
                    print(f"Using demo clip: {path}")
                if not path.exists():
                    print(f"File not found: {path}")
                    continue
                top_k = _prompt_int("Top-K", args.top_k)
                mode = _prompt_choice("Mode (video/frame)", ("video", "frame"), "video")
                payload = service.reverse_search(
                    video_path=str(path),
                    top_k=top_k,
                    mode=mode,
                )
                _print_payload_summary(payload)
                _print_results("Video-to-text retrieval results", payload.get("results") or [])

                results = payload.get("results") or []
                if results:
                    summarize = _prompt_choice("Generate summary explanation? (y/n)", ("y", "n"), "y") == "y"
                    if summarize:
                        rag = service.rag(
                            query=f"Descrie clipul încărcat: {path.name}",
                            results=results[:5],
                            model_name=args.model,
                        )
                        print()
                        print("Video-to-text summary")
                        print("=====================")
                        print(rag.get("answer") or rag.get("explanation") or rag.get("summary") or rag)
                service.release_resources()
                continue

            print("Unknown choice. Please pick 1, 2, or 3.")
    finally:
        try:
            if service is not None:
                service.release_resources()
        except Exception:
            pass


if __name__ == "__main__":
    main()
