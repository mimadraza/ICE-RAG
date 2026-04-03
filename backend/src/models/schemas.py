from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pydantic import BaseModel


# ── Pydantic request/response schemas ──────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    chunking: str = "recursive"       # fixed | recursive | semantic
    retrieval: str = "hybrid"         # semantic | hybrid
    rerank: bool = True
    retrieve_k: int = 10
    final_k: int = 5


class ExperimentRequest(BaseModel):
    queries: list[str]
    chunking: str = "recursive"
    retrieval: str = "hybrid"
    rerank: bool = True


# ── Internal data containers ────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    chunk_id: str
    text: str
    source: str
    score: float
    retrieval_type: str
    strategy: str
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    query: str
    answer: str
    retrieved_docs: list[dict]
    retrieval_time: float
    generation_time: float
    total_time: float
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_docs": self.retrieved_docs,
            "retrieval_time": round(self.retrieval_time, 4),
            "generation_time": round(self.generation_time, 4),
            "total_time": round(self.total_time, 4),
            "config": self.config,
        }


@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: float
    relevancy: float
    claims: list[str]
    claim_verification: list[dict]
    alt_questions: list[str]
    relevancy_scores: list[float]
    retrieval_time: float
    generation_time: float
    total_time: float
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
