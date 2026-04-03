import time
from retrievers import SemanticRetriever, HybridRetriever
from rerankers import CrossEncoderReranker
from generator import generate_answer


class RAGPipeline:
    def __init__(self, strategy: str, retrieval: str, rerank: bool):
        if retrieval == "semantic":
            self.retriever = SemanticRetriever(strategy)
        elif retrieval == "hybrid":
            self.retriever = HybridRetriever(strategy)
        else:
            raise ValueError("retrieval must be semantic or hybrid")

        self.reranker = CrossEncoderReranker() if rerank else None

    def run(self, query: str, retrieve_k: int = 10, final_k: int = 5):
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

        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": docs,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
        }