from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Env ───────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV)

# Allow imports from Test_pipeline directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from evaluators import extract_claims, verify_claims, RelevancyScorer
from generator import generate_answer
from rerankers import CrossEncoderReranker

try:
    from pinecone import Pinecone
except ImportError:
    raise ImportError("Run: pip install pinecone")

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX    = "rag-index"
NAMESPACE         = "recursive"

BASE_DIR          = Path(__file__).resolve().parents[2]
CHUNK_CSV         = BASE_DIR / "data" / "chunking" / "chunks_recursive.csv"
REPORT_DIR        = BASE_DIR / "data" / "experiments"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVE_K        = 10
FINAL_K           = 5

# ── Fixed test set (15 queries) ───────────────────────────────────────────────
TEST_QUERIES = [
    {"id":  1, "query": "Can ICE enter a home with only an administrative warrant?"},
    {"id":  2, "query": "What rights does a person have during immigration questioning?"},
    {"id":  3, "query": "Can a person remain silent if approached by immigration officers?"},
    {"id":  4, "query": "What should I do if ICE comes to my door without a warrant?"},
    {"id":  5, "query": "What is the difference between a judicial warrant and an administrative warrant?"},
    {"id":  6, "query": "Can ICE arrest someone without a warrant in a public place?"},
    {"id":  7, "query": "What are my rights if I am detained by immigration officers?"},
    {"id":  8, "query": "Do I have to show identification to immigration officers?"},
    {"id":  9, "query": "Can ICE question someone about their immigration status at a workplace?"},
    {"id": 10, "query": "What happens if I open the door for ICE agents without a warrant?"},
    {"id": 11, "query": "Can an employer allow ICE into the workplace without a warrant?"},
    {"id": 12, "query": "What rights do undocumented immigrants have in the United States?"},
    {"id": 13, "query": "Does signing an ICE form waive my right to a hearing?"},
    {"id": 14, "query": "Can ICE deport someone who has a pending immigration case?"},
    {"id": 15, "query": "What is a know-your-rights card and how does it help during ICE encounters?"},
]

# ── Retrieval helpers ─────────────────────────────────────────────────────────

def build_bm25(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].reset_index(drop=True)
    meta = df.to_dict(orient="records")
    tokenized = [str(m["text"]).lower().split() for m in meta]
    return BM25Okapi(tokenized), meta


def reciprocal_rank_fusion(lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    for results in lists:
        for rank, doc in enumerate(results, start=1):
            cid = doc.get("chunk_id")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in doc_map:
                doc_map[cid] = doc
    merged = [{**doc_map[c], "rrf_score": s} for c, s in scores.items()]
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


class HybridPineconeRetriever:
    """Pinecone semantic + BM25 CSV, fused with RRF."""

    def __init__(self):
        if not PINECONE_API_KEY:
            raise EnvironmentError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self._pine = pc.Index(PINECONE_INDEX)
        self._model = SentenceTransformer(EMBED_MODEL)
        self._bm25, self._meta = build_bm25(CHUNK_CSV)

    def _semantic(self, query: str, top_k: int) -> list[dict]:
        emb = self._model.encode([query], normalize_embeddings=True)
        resp = self._pine.query(
            vector=emb[0].tolist(),
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True,
        )
        results = []
        for m in resp.matches:
            doc = dict(m.metadata) if m.metadata else {}
            doc["chunk_id"] = m.id
            doc["score"] = float(m.score)
            results.append(doc)
        return results

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        scores = self._bm25.get_scores(query.lower().split())
        idxs = np.argsort(scores)[::-1][:top_k]
        return [{**self._meta[i], "score": float(scores[i])} for i in idxs]

    def retrieve(self, query: str, top_k: int = RETRIEVE_K) -> list[dict]:
        pool = max(top_k * 3, 30)
        sem  = self._semantic(query, pool)
        bm25 = self._bm25_retrieve(query, pool)
        return reciprocal_rank_fusion([sem, bm25])[:top_k]


# ── Per-query evaluation ──────────────────────────────────────────────────────

def run_and_evaluate(
    query: str,
    retriever: HybridPineconeRetriever,
    reranker: CrossEncoderReranker,
    rel_scorer: RelevancyScorer,
) -> dict:
    # Retrieval
    t0 = time.perf_counter()
    docs = retriever.retrieve(query, top_k=RETRIEVE_K)
    docs = reranker.rerank(query, docs, top_k=FINAL_K)
    retrieval_time = time.perf_counter() - t0

    # Generation
    t1 = time.perf_counter()
    answer = generate_answer(query, docs)
    generation_time = time.perf_counter() - t1

    # Context for faithfulness
    context = "\n\n".join(d.get("text", "") for d in docs)

    # Faithfulness
    claims = extract_claims(answer)
    faith_score, verifications = verify_claims(claims, context)

    # Relevancy
    alt_questions = rel_scorer.generate_alt_questions(answer)
    rel_score, rel_scores = rel_scorer.score(query, alt_questions)

    return {
        "query":             query,
        "answer":            answer,
        "retrieved_docs":    docs,
        "retrieval_time":    round(retrieval_time, 3),
        "generation_time":   round(generation_time, 3),
        "total_time":        round(retrieval_time + generation_time, 3),
        "claims":            claims,
        "claim_verification": verifications,
        "faithfulness":      round(faith_score, 4),
        "alt_questions":     alt_questions,
        "relevancy_scores":  [round(s, 4) for s in rel_scores],
        "relevancy":         round(rel_score, 4),
    }


# ── Report generation ─────────────────────────────────────────────────────────

def _supported(v: dict) -> str:
    return "✓" if v.get("label") == "supported" else "✗"


def build_text_report(results: list[dict], detail_ids: list[int]) -> str:
    lines = []
    sep  = "=" * 70
    sep2 = "-" * 70

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel   = sum(r["relevancy"]    for r in results) / len(results)

    lines += [
        sep,
        "  LLM-AS-A-JUDGE EVALUATION REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Config    : recursive | hybrid (Pinecone + BM25) | rerank=True",
        f"  Queries   : {len(results)}",
        sep,
        "",
        "AGGREGATE METRICS",
        f"  Average Faithfulness : {avg_faith:.4f}  ({avg_faith*100:.1f}%)",
        f"  Average Relevancy    : {avg_rel:.4f}",
        "",
        sep2,
        "QUERY SUMMARY",
        sep2,
        f"  {'ID':>3}  {'Faithfulness':>12}  {'Relevancy':>10}  {'Claims':>6}  Query",
        sep2,
    ]

    for r in results:
        n_claims = len(r["claims"])
        lines.append(
            f"  {r['id']:>3}  {r['faithfulness']:>12.4f}  {r['relevancy']:>10.4f}"
            f"  {n_claims:>6}  {r['query'][:55]}"
        )

    lines += ["", sep]

    # ── Detailed examples ──────────────────────────────────────────────────────
    detail_map = {r["id"]: r for r in results}
    lines.append("  DETAILED EXAMPLES")
    lines.append(sep)

    for qid in detail_ids:
        r = detail_map.get(qid)
        if not r:
            continue
        lines += [
            "",
            f"  [QUERY {r['id']}]  {r['query']}",
            sep2,
            "",
            "  ANSWER:",
        ]
        # Word-wrap answer at ~65 chars
        words, line_buf = r["answer"].split(), []
        for w in words:
            line_buf.append(w)
            if len(" ".join(line_buf)) > 65:
                lines.append("    " + " ".join(line_buf[:-1]))
                line_buf = [w]
        if line_buf:
            lines.append("    " + " ".join(line_buf))

        lines += ["", "  FAITHFULNESS"]
        if r["claims"]:
            for v in r["claim_verification"]:
                icon   = _supported(v)
                label  = v.get("label", "?").upper()
                reason = v.get("reason", "")[:80]
                claim  = v.get("claim", "")[:70]
                lines.append(f"    {icon} [{label}]  {claim}")
                lines.append(f"         Reason: {reason}")
        else:
            lines.append("    (no claims extracted)")

        lines.append(f"  → Faithfulness Score : {r['faithfulness']:.4f}  ({r['faithfulness']*100:.1f}%)")

        lines += ["", "  RELEVANCY — ALTERNATIVE QUESTIONS"]
        for i, (aq, s) in enumerate(zip(r["alt_questions"], r["relevancy_scores"]), 1):
            lines.append(f"    Q{i} (sim={s:.4f}) : {aq}")
        lines.append(f"  → Average Relevancy  : {r['relevancy']:.4f}")
        lines.append("")
        lines.append(sep2)

    lines += ["", sep, "END OF REPORT", sep]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Initialising components...")
    retriever  = HybridPineconeRetriever()
    reranker   = CrossEncoderReranker()
    rel_scorer = RelevancyScorer()
    print(f"Running evaluation on {len(TEST_QUERIES)} queries...\n")

    results = []
    for item in TEST_QUERIES:
        qid   = item["id"]
        query = item["query"]
        print(f"  [{qid:02d}/{len(TEST_QUERIES)}] {query[:65]}")

        result = run_and_evaluate(query, retriever, reranker, rel_scorer)
        result["id"] = qid
        results.append(result)

        print(
            f"         faith={result['faithfulness']:.3f}"
            f"  rel={result['relevancy']:.3f}"
            f"  claims={len(result['claims'])}"
            f"  t={result['total_time']:.1f}s"
        )

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = REPORT_DIR / "llm_judge_results.json"

    # Serialise: strip retrieved_docs text to keep file manageable
    json_safe = []
    for r in results:
        rec = {k: v for k, v in r.items() if k != "retrieved_docs"}
        rec["sources"] = [
            {"chunk_id": d.get("chunk_id"), "doc_id": d.get("doc_id")}
            for d in r["retrieved_docs"]
        ]
        json_safe.append(rec)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

    # ── Save text report ───────────────────────────────────────────────────────
    # Show detailed breakdown for queries 1, 7, 13
    detail_ids = [1, 7, 13]
    report_text = build_text_report(results, detail_ids)

    txt_path = REPORT_DIR / "llm_judge_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel   = sum(r["relevancy"]    for r in results) / len(results)
    print(f"  FINAL  Avg Faithfulness: {avg_faith:.4f}  ({avg_faith*100:.1f}%)")
    print(f"  FINAL  Avg Relevancy   : {avg_rel:.4f}")
    print("=" * 55)
    print(f"\nJSON report  → {json_path.relative_to(BASE_DIR)}")
    print(f"Text report  → {txt_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
