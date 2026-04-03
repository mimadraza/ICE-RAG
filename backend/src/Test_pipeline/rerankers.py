from sentence_transformers import CrossEncoder
from config import RERANK_MODEL


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5):
        pairs = [[query, d["text"]] for d in docs]
        scores = self.model.predict(pairs)

        rescored = []
        for d, s in zip(docs, scores):
            rescored.append({**d, "rerank_score": float(s)})

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return rescored[:top_k]