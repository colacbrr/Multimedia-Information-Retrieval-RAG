from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


EMBEDDING_DIM = 512


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def encode_image_paths(
    model,
    preprocess,
    device: str,
    image_paths: Iterable[str | Path],
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    paths = [str(Path(path)) for path in image_paths]
    if not paths:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    model.eval()
    embeddings: list[np.ndarray] = []
    valid_paths: list[str] = []
    with torch.no_grad():
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset : offset + batch_size]
            batch_images = []
            batch_valid_paths = []
            for path in batch_paths:
                try:
                    image = preprocess(Image.open(path).convert("RGB"))
                    batch_images.append(image)
                    batch_valid_paths.append(path)
                except Exception:
                    continue
            if not batch_images:
                continue
            tensor = torch.stack(batch_images).to(device)
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
            valid_paths.extend(batch_valid_paths)

    if not embeddings:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    return np.vstack(embeddings).astype(np.float32), valid_paths


def encode_texts(
    model,
    device: str,
    texts: Iterable[str],
    clip_tokenize,
    batch_size: int = 128,
    progress_cb=None,
) -> np.ndarray:
    items = [str(text) for text in texts if str(text).strip()]
    if not items:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model.eval()
    embeddings: list[np.ndarray] = []
    processed = 0
    with torch.no_grad():
        for offset in range(0, len(items), batch_size):
            batch_texts = items[offset : offset + batch_size]
            tokens = clip_tokenize(batch_texts).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            batch_embeddings = features.cpu().numpy().astype(np.float32)
            embeddings.append(batch_embeddings)
            processed += len(batch_texts)
            if progress_cb is not None:
                progress_cb(processed, len(items))
    if not embeddings:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32)


def aggregate_embeddings(vectors: np.ndarray, method: str = "mean") -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    method = (method or "mean").lower()
    if method == "max":
        agg = vectors.max(axis=0, keepdims=True)
    else:
        agg = vectors.mean(axis=0, keepdims=True)
    return l2_normalize(agg)
