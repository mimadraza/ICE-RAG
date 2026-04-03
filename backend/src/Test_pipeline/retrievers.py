from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import EMBED_DIR, EMBED_MODEL


def load_strategy_data(strategy: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Load the FAISS index and metadata for a chunking strategy.

    Expected files:
      EMBED_DIR/<strategy>/index.faiss
      EMBED_DIR/<strategy>/metadata.pkl
    """
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
        raise ValueError(f"Expected metadata to be a list, got {type(meta)}")

    return index, meta


def _normalize_query_embedding(vec: np.ndarray) -> np.ndarray:
    """
    Normalize query vector for cosine-style similarity search in FAISS.
    Assumes the index was built with normalized document embeddings.
    """
    vec = np.asarray(vec, dtype="float32")
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)

    faiss.normalize_L2(vec)
    return vec


class SemanticRetriever:
    """
    Semantic retrieval backed by a FAISS index.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index, self.meta = load_strategy_data(strategy)
        self.model = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        distances, indices = self.index.search(query_emb, top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx >= len(self.meta):
                continue

            results.append({
                **self.meta[idx],
                "score": float(score),
                "retrieval_type": "semantic",
                "strategy": self.strategy,
            })

        return results


class BM25Retriever:
    """
    Lexical retrieval over the metadata text field.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        _, self.meta = load_strategy_data(strategy)

        self.texts = [str(m.get("text", "")) for m in self.meta]
        self.tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in idxs:
            results.append({
                **self.meta[idx],
                "score": float(scores[idx]),
                "retrieval_type": "bm25",
                "strategy": self.strategy,
            })

        return results


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc

    merged = []
    for chunk_id, score in fused_scores.items():
        merged.append({
            **doc_map[chunk_id],
            "rrf_score": float(score),
            "retrieval_type": "hybrid_rrf",
        })

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


class HybridRetriever:
    """
    Hybrid retriever using semantic + BM25, combined with RRF.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.semantic = SemanticRetriever(strategy)
        self.bm25 = BM25Retriever(strategy)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        semantic_results = self.semantic.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)

        fused = reciprocal_rank_fusion([semantic_results, bm25_results])
        return fused[:top_k]