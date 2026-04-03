from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import EMBED_DIR, EMBED_MODEL

_model_cache: dict[str, SentenceTransformer] = {}
_data_cache: dict[str, tuple] = {}


def _get_model() -> SentenceTransformer:
    if EMBED_MODEL not in _model_cache:
        _model_cache[EMBED_MODEL] = SentenceTransformer(EMBED_MODEL)
    return _model_cache[EMBED_MODEL]


def load_strategy_data(strategy: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
    if strategy in _data_cache:
        return _data_cache[strategy]

    strategy_dir = Path(EMBED_DIR) / strategy
    index_path = strategy_dir / "index.faiss"
    meta_path = strategy_dir / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if not isinstance(meta, list):
        raise ValueError(f"Expected list metadata, got {type(meta)}")

    _data_cache[strategy] = (index, meta)
    return index, meta


class SemanticRetriever:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index, self.meta = load_strategy_data(strategy)
        self.model = _get_model()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.meta):
                continue
            results.append({
                **self.meta[idx],
                "score": float(score),
                "retrieval_type": "semantic",
                "strategy": self.strategy,
            })
        return results


class BM25Retriever:
    def __init__(self, strategy: str):
        self.strategy = strategy
        _, self.meta = load_strategy_data(strategy)
        self.texts = [str(m.get("text", "")) for m in self.meta]
        self.tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:top_k]

        return [{
            **self.meta[idx],
            "score": float(scores[idx]),
            "retrieval_type": "bm25",
            "strategy": self.strategy,
        } for idx in idxs]


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]], k: int = 60
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc

    merged = [
        {**doc_map[cid], "rrf_score": float(score), "retrieval_type": "hybrid_rrf"}
        for cid, score in fused_scores.items()
    ]
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


class HybridRetriever:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.semantic = SemanticRetriever(strategy)
        self.bm25 = BM25Retriever(strategy)

    def retrieve(self, query: str, top_k: int = 10, candidate_k: int = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        # Always fetch at least 3x top_k per source so RRF has enough candidates to
        # distinguish good from mediocre results before truncating.
        pool = max(top_k * 3, 30) if candidate_k is None else candidate_k
        sem = self.semantic.retrieve(query, top_k=pool)
        bm25 = self.bm25.retrieve(query, top_k=pool)
        fused = reciprocal_rank_fusion([sem, bm25])
        return fused[:top_k]
