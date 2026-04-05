"""
Build Pinecone vector index from existing FAISS indices.

Usage:
    python build_pinecone.py

Reads local FAISS indices + metadata.pkl for each strategy (fixed / recursive / semantic),
then upserts all vectors into a single Pinecone serverless index using namespaces.

Requirements:
    pip install pinecone
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    raise ImportError("Run: pip install pinecone")

# ── Config ────────────────────────────────────────────────────────────────────

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME       = "rag-index"
DIMENSION        = 384          # all-MiniLM-L6-v2 output dim
METRIC           = "cosine"
CLOUD            = "aws"
REGION           = "us-east-1"

STRATEGIES = ["recursive"]

BASE_DIR   = Path(__file__).resolve().parent
EMBED_DIR  = BASE_DIR / "data" / "embeddings"

BATCH_SIZE = 100   # Pinecone recommended upsert batch size

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_faiss_and_meta(strategy: str) -> tuple[np.ndarray, list[dict]]:
    """Return (vectors ndarray, metadata list) for a strategy."""
    strategy_dir = EMBED_DIR / strategy
    index_path   = strategy_dir / "index.faiss"
    meta_path    = strategy_dir / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    index = faiss.read_index(str(index_path))
    n = index.ntotal
    if n == 0:
        raise ValueError(f"FAISS index for '{strategy}' is empty.")

    # IndexFlatIP stores raw vectors → reconstruct them all
    vectors = np.zeros((n, DIMENSION), dtype="float32")
    index.reconstruct_n(0, n, vectors)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if len(meta) != n:
        raise ValueError(
            f"Metadata length {len(meta)} != FAISS vector count {n} for '{strategy}'"
        )

    return vectors, meta


def meta_to_pinecone(record: dict) -> dict:
    """Keep only Pinecone-safe scalar metadata fields."""
    safe = {}
    for k, v in record.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            safe[k] = v
    return safe


def upsert_strategy(index, strategy: str, vectors: np.ndarray, meta: list[dict]) -> int:
    """Upsert all vectors for one strategy into its namespace. Returns vector count."""
    namespace = strategy
    total     = len(meta)
    upserted  = 0

    print(f"\n  [{strategy.upper()}] Upserting {total:,} vectors into namespace='{namespace}'...")

    for start in range(0, total, BATCH_SIZE):
        batch_vecs = vectors[start : start + BATCH_SIZE]
        batch_meta = meta[start : start + BATCH_SIZE]

        records = []
        for vec, record in zip(batch_vecs, batch_meta):
            chunk_id = str(record.get("chunk_id", f"{strategy}_{start}"))
            records.append({
                "id":     chunk_id,
                "values": vec.tolist(),
                "metadata": meta_to_pinecone(record),
            })

        index.upsert(vectors=records, namespace=namespace)
        upserted += len(records)

        if upserted % 500 == 0 or upserted == total:
            print(f"    {upserted:,} / {total:,} upserted")

    return upserted


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' ({DIMENSION}d, {METRIC})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Wait for index to be ready
        while True:
            info = pc.describe_index(INDEX_NAME)
            if info.status.get("ready", False):
                break
            print("  Waiting for index to be ready...")
            time.sleep(3)
        print("  Index ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists — upserting into it.")

    pine_index = pc.Index(INDEX_NAME)

    total_upserted = 0
    t0 = time.time()

    for strategy in STRATEGIES:
        vectors, meta = load_faiss_and_meta(strategy)
        count = upsert_strategy(pine_index, strategy, vectors, meta)
        total_upserted += count

    elapsed = time.time() - t0
    print(f"\nDone — {total_upserted:,} vectors upserted across {len(STRATEGIES)} namespaces in {elapsed:.1f}s")

    # Quick stats
    stats = pine_index.describe_index_stats()
    print("\nIndex stats:")
    for ns, ns_stats in stats.namespaces.items():
        print(f"  namespace={ns!r:20s}  vector_count={ns_stats.vector_count:,}")


if __name__ == "__main__":
    main()
