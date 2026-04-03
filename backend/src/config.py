from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHUNK_DIR = DATA_DIR / "chunking"
EMBED_DIR = DATA_DIR / "embeddings"
EXPERIMENT_DIR = DATA_DIR / "experiments"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_FILES = {
    "fixed": CHUNK_DIR / "chunks_fixed.csv",
    "recursive": CHUNK_DIR / "chunks_recursive.csv",
    "semantic": CHUNK_DIR / "chunks_semantic.csv",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SEMANTIC_THRESHOLD = 0.85

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEN_MODEL = "llama-3.3-70b-versatile"

# Default best config based on experiments: recursive + hybrid + rerank
DEFAULT_CHUNKING = "recursive"
DEFAULT_RETRIEVAL = "hybrid"
DEFAULT_RERANK = True
