from __future__ import annotations

import os
from typing import Any

from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL

_model_cache: dict[str, SentenceTransformer] = {}
_index_cache: dict = {}   # lazily holds the Pinecone Index object

PINECONE_INDEX_NAME = "rag-index"


def _get_model() -> SentenceTransformer:
    if EMBED_MODEL not in _model_cache:
        _model_cache[EMBED_MODEL] = SentenceTransformer(EMBED_MODEL)
    return _model_cache[EMBED_MODEL]


def _get_pine_index():
    """Return a cached Pinecone Index object."""
    if "index" not in _index_cache:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("Run: pip install pinecone")

        api_key = os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set")

        pc = Pinecone(api_key=api_key)
        _index_cache["index"] = pc.Index(PINECONE_INDEX_NAME)

    return _index_cache["index"]


class PineconeRetriever:
    """Semantic retriever backed by Pinecone (drop-in replacement for SemanticRetriever)."""

    def __init__(self, strategy: str):
        self.strategy  = strategy
        self.namespace = strategy
        self.model     = _get_model()
        self._index    = _get_pine_index()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        response = self._index.query(
            vector=query_emb[0].tolist(),
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        results = []
        for match in response.matches:
            doc = dict(match.metadata) if match.metadata else {}
            doc["chunk_id"]      = match.id
            doc["score"]         = float(match.score)
            doc["retrieval_type"] = "semantic"
            doc["strategy"]      = self.strategy
            results.append(doc)

        return results
