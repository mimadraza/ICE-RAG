"""
View layer — formats controller output into API response shapes.
All responses go through these helpers so the shape stays consistent.
"""
from typing import Any


def success(data: Any, message: str = "ok") -> dict:
    return {"status": "success", "message": message, "data": data}


def error(message: str, code: int = 500) -> dict:
    return {"status": "error", "message": message, "code": code}


def query_view(result: dict) -> dict:
    """Shape a query result for the API consumer."""
    return success({
        "query": result["query"],
        "answer": result["answer"],
        "config": result.get("config", {}),
        "timing": {
            "retrieval_ms": round(result["retrieval_time"] * 1000, 1),
            "generation_ms": round(result["generation_time"] * 1000, 1),
            "total_ms": round(result["total_time"] * 1000, 1),
        },
        "sources": [
            {
                "chunk_id": d.get("chunk_id", ""),
                "text": d.get("text", "")[:300],  # truncate for UI
                "source": d.get("source", ""),
                "score": d.get("rerank_score") or d.get("rrf_score") or d.get("score", 0),
                "retrieval_type": d.get("retrieval_type", ""),
            }
            for d in result.get("retrieved_docs", [])
        ],
    })


def experiment_summary_view(data: dict) -> dict:
    return success(data)


def experiment_run_view(data: dict) -> dict:
    return success(data)


def info_view(data: dict) -> dict:
    return success(data)
