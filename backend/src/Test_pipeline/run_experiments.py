import json
import pandas as pd
from pathlib import Path

from config import EXPERIMENT_DIR
from pipeline import RAGPipeline
from retrievers import SemanticRetriever, HybridRetriever
from rerankers import CrossEncoderReranker
from evaluators import evaluate_run

SETTINGS = [
    {"chunking": "fixed", "retrieval": "semantic", "rerank": False},
    {"chunking": "fixed", "retrieval": "hybrid", "rerank": True},
    {"chunking": "recursive", "retrieval": "semantic", "rerank": False},
    {"chunking": "recursive", "retrieval": "hybrid", "rerank": True},
    {"chunking": "semantic", "retrieval": "semantic", "rerank": False},
    {"chunking": "semantic", "retrieval": "hybrid", "rerank": True},
]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_FILE = BASE_DIR / "data" / "eval_queries.json"
RUN_RESULTS_FILE = EXPERIMENT_DIR / "run_results.csv"
DETAILS_FILE = EXPERIMENT_DIR / "detailed_runs.json"
SUMMARY_FILE = EXPERIMENT_DIR / "summary_report.csv"


def load_eval_queries():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    queries = load_eval_queries()
    rows = []
    details = []

    for cfg in SETTINGS:
        pipe = RAGPipeline(
            strategy=cfg["chunking"],
            retrieval=cfg["retrieval"],
            rerank=cfg["rerank"],
        )

        for q in queries:
            run = pipe.run(q["query"], retrieve_k=10, final_k=5)
            eval_result = evaluate_run(q["query"], run["answer"], run["retrieved_docs"])

            row = {
                "query_id": q["id"],
                "query": q["query"],
                "chunking": cfg["chunking"],
                "retrieval": cfg["retrieval"],
                "rerank": cfg["rerank"],
                "faithfulness": eval_result["faithfulness"],
                "relevancy": eval_result["relevancy"],
                "retrieval_time": run["retrieval_time"],
                "generation_time": run["generation_time"],
                "total_time": run["total_time"],
            }
            rows.append(row)

            details.append({
                **row,
                "answer": run["answer"],
                "retrieved_docs": run["retrieved_docs"],
                "claims": eval_result["claims"],
                "claim_verification": eval_result["claim_verification"],
                "alt_questions": eval_result["alt_questions"],
                "relevancy_scores": eval_result["relevancy_scores"],
            })

            print(
                f"{cfg['chunking']:10} | {cfg['retrieval']:8} | rerank={cfg['rerank']} "
                f"| q={q['id']} | faith={eval_result['faithfulness']:.2f} "
                f"| rel={eval_result['relevancy']:.2f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(RUN_RESULTS_FILE, index=False, encoding="utf-8")

    with open(DETAILS_FILE, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    summary = (
        df.groupby(["chunking", "retrieval", "rerank"], as_index=False)
        .agg({
            "faithfulness": "mean",
            "relevancy": "mean",
            "retrieval_time": "mean",
            "generation_time": "mean",
            "total_time": "mean",
        })
        .sort_values(["faithfulness", "relevancy"], ascending=False)
    )
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()