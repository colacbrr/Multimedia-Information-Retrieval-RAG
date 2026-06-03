from __future__ import annotations

from typing import Any


def format_result_context(results: list[dict[str, Any]]) -> list[str]:
    lines = []
    for idx, item in enumerate(results, start=1):
        video_file = item.get("video_file") or item.get("file_name") or "unknown"
        score = item.get("score", 0.0)
        timestamp = item.get("best_timestamp")
        timestamp_text = f"{float(timestamp):.2f}s" if isinstance(timestamp, (int, float)) else "necunoscut"
        caption = item.get("caption") or item.get("caption_context") or "fara caption"
        lines.append(
            f"{idx}. {video_file} | score {float(score):.4f} | timestamp {timestamp_text} | caption/context: {caption}"
        )
    return lines


def build_video_rag_prompt(query: str, results: list[dict[str, Any]]) -> str:
    context = "\n".join(format_result_context(results))
    return (
        "Raspunde concis in romana, folosind doar contextul dat.\n"
        f"Query: {query}\n"
        f"Rezultate video:\n{context}\n"
        "Explica de ce rezultatele sunt relevante. Nu inventa detalii care nu apar in context."
    )


def build_fallback_video_summary(
    query: str,
    results: list[dict[str, Any]],
    reason: str | None = None,
) -> str:
    if not results:
        lines = [
            "Nu au fost gasite rezultate relevante pentru sumarizare.",
            f"Query: {query}",
        ]
        if reason:
            lines.append(f"Motiv fallback: {reason}")
        return "\n".join(lines)

    top = results[:5]
    captions = []
    sources = []
    timestamps = []
    for item in top:
        caption = item.get("caption") or item.get("caption_context") or "fara caption"
        captions.append(str(caption))
        source = item.get("source") or item.get("source_bucket") or item.get("relative_path") or item.get("video_file")
        if source:
            sources.append(str(source))
        timestamp = item.get("best_timestamp") or item.get("timestamp")
        if isinstance(timestamp, (int, float)):
            timestamps.append(f"{float(timestamp):.2f}s")

    unique_captions = list(dict.fromkeys(captions))
    unique_sources = list(dict.fromkeys(sources))
    unique_timestamps = list(dict.fromkeys(timestamps))
    main_caption = unique_captions[0] if unique_captions else "fara caption"

    lines = [
        "Sumar fallback generat local, fara Ollama.",
        f"Query: {query}",
        f"Context dominant: {main_caption}",
        f"Concluzie: rezultatele sugereaza ca acest clip este cel mai probabil despre {main_caption.lower().rstrip('.')}.",
    ]
    if unique_sources:
        lines.append("Surse potrivite: " + ", ".join(unique_sources[:5]))
    if unique_timestamps:
        lines.append("Timestamp-uri utile: " + ", ".join(unique_timestamps[:5]))
    if len(unique_captions) > 1:
        lines.append("Alte indicii: " + " | ".join(unique_captions[1:4]))
    if reason:
        lines.append(f"Motiv fallback: {reason}")
    return "\n".join(lines)


def build_reverse_prompt(video_file: str, results: list[dict[str, Any]]) -> str:
    context = "\n".join(format_result_context(results))
    return (
        "Raspunde concis in romana, folosind doar contextul dat.\n"
        f"Video: {video_file}\n"
        f"Rezultate text:\n{context}\n"
        "Descrie pe scurt ce texte sunt cele mai apropiate si de ce."
    )
