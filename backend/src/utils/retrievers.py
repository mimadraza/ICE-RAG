from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import CHUNK_FILES
from src.utils.pinecone_retriever import PineconeRetriever

_bm25_cache: dict[str, tuple] = {}


def _load_bm25_data(strategy: str) -> tuple[list[str], list[dict]]:
    if strategy in _bm25_cache:
        return _bm25_cache[strategy]

    csv_path = CHUNK_FILES.get(strategy)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"Chunk CSV not found for strategy '{strategy}': {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].reset_index(drop=True)
    texts = df["text"].tolist()
    meta  = df.to_dict(orient="records")

    _bm25_cache[strategy] = (texts, meta)
    return texts, meta


class BM25Retriever:
    def __init__(self, strategy: str):
        self.strategy = strategy
        texts, self.meta = _load_bm25_data(strategy)
        self.tokenized = [t.lower().split() for t in texts]
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
        self.semantic = PineconeRetriever(strategy)
        self.bm25     = BM25Retriever(strategy)

    def retrieve(self, query: str, top_k: int = 10, candidate_k: int = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        pool = max(top_k * 3, 30) if candidate_k is None else candidate_k
        sem  = self.semantic.retrieve(query, top_k=pool)
        bm25 = self.bm25.retrieve(query, top_k=pool)
        fused = reciprocal_rank_fusion([sem, bm25])
        return fused[:top_k]
