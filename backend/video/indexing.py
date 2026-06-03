from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(
    vectors: np.ndarray,
    index_type: str = "flat",
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 40,
) -> object:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Expected a 2D array for FAISS index construction.")
    if vectors.shape[0] == 0:
        raise ValueError("Cannot build a FAISS index from an empty vector set.")
    dim = vectors.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT
    index_type = (index_type or "flat").lower()
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, metric)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = hnsw_ef_construction
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: object, path: str | Path) -> None:
    faiss.write_index(index, str(Path(path)))


def load_index(path: str | Path) -> object:
    return faiss.read_index(str(Path(path)))


def search_index(index: object, query_vector: np.ndarray, top_k: int = 5):
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim == 1:
        query_vector = query_vector[None, :]
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]
