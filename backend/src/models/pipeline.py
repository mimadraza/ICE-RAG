import time
from src.utils.retrievers import SemanticRetriever, HybridRetriever
from src.utils.rerankers import CrossEncoderReranker
from src.utils.generator import generate_answer
from src.models.schemas import QueryResult

_pipeline_cache: dict[str, "RAGPipeline"] = {}


class RAGPipeline:
    """Core RAG pipeline — retriever + optional reranker + generator."""

    def __init__(self, strategy: str, retrieval: str, rerank: bool):
        if retrieval == "semantic":
            self.retriever = SemanticRetriever(strategy)
        elif retrieval == "hybrid":
            self.retriever = HybridRetriever(strategy)
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval}")

        self.reranker = CrossEncoderReranker() if rerank else None
        self.config = {"chunking": strategy, "retrieval": retrieval, "rerank": rerank}

    def run(self, query: str, retrieve_k: int = 10, final_k: int = 5) -> QueryResult:
        t0 = time.perf_counter()
        docs = self.retriever.retrieve(query, top_k=retrieve_k)
        retrieval_time = time.perf_counter() - t0

        if self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=final_k)
        else:
            docs = docs[:final_k]

        t1 = time.perf_counter()
        answer = generate_answer(query, docs)
        generation_time = time.perf_counter() - t1

        return QueryResult(
            query=query,
            answer=answer,
            retrieved_docs=docs,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=retrieval_time + generation_time,
            config=self.config,
        )


def get_pipeline(strategy: str, retrieval: str, rerank: bool) -> RAGPipeline:
    """Return a cached pipeline instance for the given config."""
    key = f"{strategy}:{retrieval}:{rerank}"
    if key not in _pipeline_cache:
        _pipeline_cache[key] = RAGPipeline(strategy, retrieval, rerank)
    return _pipeline_cache[key]
