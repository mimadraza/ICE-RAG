import json
import pickle
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import faiss
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")


BASE_DIR      = Path(__file__).resolve().parent.parent.parent
CHUNKING_DIR  = BASE_DIR / "Test" / "chunking"
EMBEDDINGS_DIR = BASE_DIR / "Test" / "embeddings"
REPORT_FILE   = EMBEDDINGS_DIR / "build_report.txt"

STRATEGIES = ["fixed", "recursive", "semantic"]

CSV_MAP = {
    "fixed":     CHUNKING_DIR / "chunks_fixed.csv",
    "recursive": CHUNKING_DIR / "chunks_recursive.csv",
    "semantic":  CHUNKING_DIR / "chunks_semantic.csv",
}


EMBEDDING_MODEL = "all-MiniLM-L6-v2"   
BATCH_SIZE      = 64                    
SMOKE_TEST_QUERY = "What rights do I have if ICE comes to my door without a warrant?"
TOP_K           = 3



def build_index(strategy: str, model: SentenceTransformer) -> dict:
    csv_path = CSV_MAP[strategy]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nChunks CSV not found: {csv_path}\n"
            f"Run chunking.py first to generate the CSV files."
        )

    out_dir = EMBEDDINGS_DIR / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Strategy: {strategy.upper()}")
    print(f"{'='*55}")

    
    df = pd.read_csv(csv_path)
   
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].reset_index(drop=True)
    print(f"  Loaded {len(df):,} chunks from {csv_path.name}")

    texts    = df["text"].tolist()
    metadata = df.to_dict(orient="records")

    print(f"  Embedding {len(texts):,} chunks in batches of {BATCH_SIZE}...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   
    )
    embed_time = time.time() - t0
    print(f"  Embedding done in {embed_time:.1f}s")

    dim   = embeddings.shape[1]           # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)        # Inner Product = cosine sim on normalized vecs
    index.add(embeddings.astype("float32"))
    print(f"  FAISS index built  — {index.ntotal:,} vectors, dim={dim}")

    faiss_path    = out_dir / "index.faiss"
    metadata_path = out_dir / "metadata.pkl"

    faiss.write_index(index, str(faiss_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"  Saved → {faiss_path.relative_to(BASE_DIR)}")
    print(f"  Saved → {metadata_path.relative_to(BASE_DIR)}")

    return {
        "strategy":    strategy,
        "num_chunks":  len(texts),
        "embed_dim":   dim,
        "embed_time_s": round(embed_time, 1),
        "index_path":  str(faiss_path.relative_to(BASE_DIR)),
        "meta_path":   str(metadata_path.relative_to(BASE_DIR)),
    }


def smoke_test(strategy: str, model: SentenceTransformer) -> list[dict]:
    """
    Load the saved FAISS index and run a single test query.
    Returns top-K results with scores.
    """
    out_dir       = EMBEDDINGS_DIR / strategy
    faiss_path    = out_dir / "index.faiss"
    metadata_path = out_dir / "metadata.pkl"

    index = faiss.read_index(str(faiss_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Embed query (must normalize to match index)
    q_emb = model.encode(
        [SMOKE_TEST_QUERY],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(q_emb, TOP_K)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        chunk = metadata[idx]
        results.append({
            "rank":     rank,
            "score":    round(float(score), 4),
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id":   chunk.get("doc_id", ""),
            "text":     chunk.get("text", "")[:300],   # first 300 chars
        })
    return results



def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")

    all_stats    = []
    smoke_results = {}

    
    for strategy in STRATEGIES:
        stats = build_index(strategy, model)
        all_stats.append(stats)

    print(f"\n{'='*55}")
    print(f"  SMOKE TEST")
    print(f"  Query: \"{SMOKE_TEST_QUERY}\"")
    print(f"{'='*55}")

    for strategy in STRATEGIES:
        print(f"\n  [{strategy.upper()}] Top {TOP_K} results:")
        results = smoke_test(strategy, model)
        smoke_results[strategy] = results
        for r in results:
            print(f"    Rank {r['rank']}  score={r['score']}  {r['doc_id']}")
            print(f"    \"{r['text'][:150]}...\"")

    # ── Save report ────────────────────────────────────────────────────────
    sep = "=" * 55
    lines = [
        sep,
        "FAISS INDEX BUILD REPORT — ICE Rights RAG Corpus",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model      : {EMBEDDING_MODEL}",
        sep,
        "",
        "Index summary:",
        f"  {'Strategy':<12} {'Chunks':>8} {'Dim':>6} {'Build time':>12}",
        "-" * 44,
        *[
            f"  {s['strategy']:<12} {s['num_chunks']:>8,} {s['embed_dim']:>6} {s['embed_time_s']:>10.1f}s"
            for s in all_stats
        ],
        "",
        sep,
        f"Smoke test query: \"{SMOKE_TEST_QUERY}\"",
        sep,
    ]

    for strategy in STRATEGIES:
        lines += [f"\n  [{strategy.upper()}]"]
        for r in smoke_results[strategy]:
            lines.append(f"    Rank {r['rank']}  score={r['score']}  {r['doc_id']}")
            lines.append(f"    \"{r['text'][:200]}\"")

    lines += [
        "",
        sep,
        "Next step: run hybrid_search.py to test BM25 + semantic retrieval",
        "           against all 3 indexes and compare retrieval quality.",
        sep,
    ]

    report_text = "\n".join(lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print(f"\n\nReport saved → {REPORT_FILE.relative_to(BASE_DIR)}")
    print("\nAll 3 FAISS indexes built successfully.")
    print("Next step: run the hybrid search + retrieval script (Stage 4).")


if __name__ == "__main__":
    main()

    