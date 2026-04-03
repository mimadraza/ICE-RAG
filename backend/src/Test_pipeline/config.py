from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_MD_DIR = PROJECT_ROOT / "clean_md"
TEST_DIR = PROJECT_ROOT / "data"
CHUNK_DIR = TEST_DIR / "chunking"
EMBED_DIR = TEST_DIR / "embeddings"
EXPERIMENT_DIR = TEST_DIR / "experiments"

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

HF_TOKEN = os.getenv("HF_TOKEN", "")
GEN_MODEL = "llama-3.1-8b-instant"
GEN_PROVIDER = "auto"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
