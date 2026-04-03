from src.models.experiments import get_available_strategies, get_best_config
from src.config import EMBED_MODEL, RERANK_MODEL, GEN_MODEL, DEFAULT_CHUNKING, DEFAULT_RETRIEVAL, DEFAULT_RERANK


class InfoController:
    """Returns system info and health status."""

    @staticmethod
    def health() -> dict:
        return {"status": "ok"}

    @staticmethod
    def info() -> dict:
        strategies = get_available_strategies()
        best = get_best_config()
        return {
            "status": "ok",
            "models": {
                "embed": EMBED_MODEL,
                "rerank": RERANK_MODEL,
                "generate": GEN_MODEL,
            },
            "available_strategies": strategies,
            "retrieval_modes": ["semantic", "hybrid"],
            "best_config": best,
            "defaults": {
                "chunking": DEFAULT_CHUNKING,
                "retrieval": DEFAULT_RETRIEVAL,
                "rerank": DEFAULT_RERANK,
            },
        }
