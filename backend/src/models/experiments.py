import csv
import json
from pathlib import Path
from src.config import EXPERIMENT_DIR, CHUNK_DIR


def load_experiment_summary() -> list[dict]:
    """Load summary_report.csv as a list of dicts."""
    path = EXPERIMENT_DIR / "summary_report.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_detailed_runs() -> list[dict]:
    """Load detailed_runs.json."""
    path = EXPERIMENT_DIR / "detailed_runs.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_chunking_report() -> str:
    """Return chunking_report.txt as a string."""
    path = CHUNK_DIR / "chunking_report.txt"
    if not path.exists():
        return ""
    return path.read_text()


def get_available_strategies() -> list[str]:
    """Return strategy names available in Pinecone."""
    return ["recursive"]


def get_best_config() -> dict:
    """Return the config with the highest faithfulness score."""
    rows = load_experiment_summary()
    if not rows:
        return {"chunking": "recursive", "retrieval": "hybrid", "rerank": True}
    best = max(rows, key=lambda r: float(r.get("faithfulness", 0)))
    return {
        "chunking": best["chunking"],
        "retrieval": best["retrieval"],
        "rerank": best["rerank"].lower() == "true",
        "faithfulness": float(best["faithfulness"]),
        "relevancy": float(best["relevancy"]),
    }
