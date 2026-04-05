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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from evaluators import extract_claims, verify_claims, RelevancyScorer
from generator import generate_answer
from rerankers import CrossEncoderReranker
from retrievers import SemanticRetriever, HybridRetriever

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "data" / "experiments"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVE_K = 10
FINAL_K    = 5

CHUNK_STRATEGIES     = ["fixed", "recursive", "semantic"]
RETRIEVAL_STRATEGIES = ["semantic", "hybrid"]

EXPERIMENTS = [
    {"chunking": chunking, "retrieval": retrieval}
    for chunking in CHUNK_STRATEGIES
    for retrieval in RETRIEVAL_STRATEGIES
]

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


# ── Retriever factory ─────────────────────────────────────────────────────────

def build_retriever(chunking: str, retrieval: str):
    if retrieval == "semantic":
        return SemanticRetriever(chunking)
    elif retrieval == "hybrid":
        return HybridRetriever(chunking)
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval!r}")


# ── Per-query evaluation ──────────────────────────────────────────────────────

def run_and_evaluate(
    query: str,
    retriever,
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
        "query":              query,
        "answer":             answer,
        "retrieved_docs":     docs,
        "retrieval_time":     round(retrieval_time, 3),
        "generation_time":    round(generation_time, 3),
        "total_time":         round(retrieval_time + generation_time, 3),
        "claims":             claims,
        "claim_verification": verifications,
        "faithfulness":       round(faith_score, 4),
        "alt_questions":      alt_questions,
        "relevancy_scores":   [round(s, 4) for s in rel_scores],
        "relevancy":          round(rel_score, 4),
    }


# ── Report helpers ────────────────────────────────────────────────────────────

def _supported(v: dict) -> str:
    return "✓" if v.get("label") == "supported" else "✗"


def build_experiment_report(
    results: list[dict],
    chunking: str,
    retrieval: str,
    detail_ids: list[int],
) -> str:
    lines = []
    sep  = "=" * 70
    sep2 = "-" * 70

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel   = sum(r["relevancy"]    for r in results) / len(results)

    lines += [
        sep,
        "  LLM-AS-A-JUDGE EVALUATION REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Config    : chunking={chunking} | retrieval={retrieval} | rerank=True",
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


def build_comparison_report(summary_rows: list[dict]) -> str:
    lines = []
    sep  = "=" * 80
    sep2 = "-" * 80

    lines += [
        sep,
        "  LLM-AS-A-JUDGE  —  EXPERIMENT COMPARISON",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Experiments : {len(summary_rows)}   |   "
        f"Chunking: {CHUNK_STRATEGIES}   |   Retrieval: {RETRIEVAL_STRATEGIES}",
        sep,
        "",
        f"  {'Chunking':>10}  {'Retrieval':>10}  {'Faithfulness':>12}  {'Relevancy':>10}"
        f"  {'Avg Time(s)':>12}",
        sep2,
    ]

    sorted_rows = sorted(summary_rows, key=lambda x: (x["avg_faithfulness"] + x["avg_relevancy"]), reverse=True)
    for row in sorted_rows:
        lines.append(
            f"  {row['chunking']:>10}  {row['retrieval']:>10}"
            f"  {row['avg_faithfulness']:>12.4f}  {row['avg_relevancy']:>10.4f}"
            f"  {row['avg_total_time']:>12.2f}"
        )

    lines += [
        "",
        sep2,
        "  Sorted by (faithfulness + relevancy) descending — best combination first.",
        sep,
        "END OF COMPARISON",
        sep,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    reranker   = CrossEncoderReranker()
    rel_scorer = RelevancyScorer()

    all_summary_rows: list[dict] = []
    detail_ids = [1, 7, 13]

    for exp in EXPERIMENTS:
        chunking  = exp["chunking"]
        retrieval = exp["retrieval"]
        label     = f"{chunking}_{retrieval}"

        print(f"\n{'='*60}")
        print(f"  Experiment: chunking={chunking}  retrieval={retrieval}")
        print(f"{'='*60}")

        try:
            retriever = build_retriever(chunking, retrieval)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        results: list[dict] = []
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

        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel   = sum(r["relevancy"]    for r in results) / len(results)
        avg_time  = sum(r["total_time"]   for r in results) / len(results)

        print(f"\n  >> {label}  faith={avg_faith:.4f}  rel={avg_rel:.4f}")

        # ── Save per-experiment JSON ───────────────────────────────────────────
        json_safe = []
        for r in results:
            rec = {k: v for k, v in r.items() if k != "retrieved_docs"}
            rec["sources"] = [
                {"chunk_id": d.get("chunk_id"), "doc_id": d.get("doc_id")}
                for d in r["retrieved_docs"]
            ]
            json_safe.append(rec)

        json_path = REPORT_DIR / f"llm_judge_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_safe, f, ensure_ascii=False, indent=2)

        # ── Save per-experiment text report ───────────────────────────────────
        report_text = build_experiment_report(results, chunking, retrieval, detail_ids)
        txt_path = REPORT_DIR / f"llm_judge_{label}.txt"
        txt_path.write_text(report_text, encoding="utf-8")

        print(f"  Saved → {json_path.name}  |  {txt_path.name}")

        all_summary_rows.append({
            "chunking":          chunking,
            "retrieval":         retrieval,
            "avg_faithfulness":  round(avg_faith, 4),
            "avg_relevancy":     round(avg_rel, 4),
            "avg_total_time":    round(avg_time, 3),
            "n_queries":         len(results),
        })

    # ── Save comparison CSV ────────────────────────────────────────────────────
    if all_summary_rows:
        csv_path = REPORT_DIR / "llm_judge_comparison.csv"
        pd.DataFrame(all_summary_rows).sort_values(
            ["avg_faithfulness", "avg_relevancy"], ascending=False
        ).to_csv(csv_path, index=False, encoding="utf-8")

        # ── Save comparison text report ────────────────────────────────────────
        cmp_txt  = build_comparison_report(all_summary_rows)
        cmp_path = REPORT_DIR / "llm_judge_comparison.txt"
        cmp_path.write_text(cmp_txt, encoding="utf-8")

        print("\n" + "=" * 60)
        print("  FINAL COMPARISON")
        print("=" * 60)
        for row in sorted(all_summary_rows, key=lambda x: x["avg_faithfulness"] + x["avg_relevancy"], reverse=True):
            print(
                f"  {row['chunking']:>10} | {row['retrieval']:>8}"
                f"  faith={row['avg_faithfulness']:.4f}"
                f"  rel={row['avg_relevancy']:.4f}"
                f"  t={row['avg_total_time']:.2f}s"
            )
        print("=" * 60)
        print(f"\nComparison CSV  → {csv_path.relative_to(BASE_DIR)}")
        print(f"Comparison TXT  → {cmp_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
