from sentence_transformers import CrossEncoder
from src.config import RERANK_MODEL

_reranker_cache = {}


def _get_reranker() -> CrossEncoder:
    if RERANK_MODEL not in _reranker_cache:
        _reranker_cache[RERANK_MODEL] = CrossEncoder(RERANK_MODEL)
    return _reranker_cache[RERANK_MODEL]


class CrossEncoderReranker:
    def __init__(self):
        self.model = _get_reranker()

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []
        pairs = [[query, d.get("text", "")] for d in docs]
        scores = self.model.predict(pairs)
        rescored = [{**d, "rerank_score": float(s)} for d, s in zip(docs, scores)]
        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return rescored[:top_k]
