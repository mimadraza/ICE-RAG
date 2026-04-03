from fastapi import HTTPException
from src.models.schemas import ExperimentRequest
from src.models.pipeline import get_pipeline
from src.utils.evaluators import evaluate_run
from src.models.experiments import (
    load_experiment_summary,
    load_detailed_runs,
    load_chunking_report,
    get_available_strategies,
    get_best_config,
)


class ExperimentController:
    """Handles experiment runs and result retrieval."""

    @staticmethod
    def get_summary() -> dict:
        """Return experiment summary data."""
        rows = load_experiment_summary()
        best = get_best_config()
        strategies = get_available_strategies()
        return {
            "results": rows,
            "best_config": best,
            "available_strategies": strategies,
        }

    @staticmethod
    def get_detailed_runs() -> list[dict]:
        return load_detailed_runs()

    @staticmethod
    def get_chunking_report() -> dict:
        return {"report": load_chunking_report()}

    @staticmethod
    def run_experiment(req: ExperimentRequest) -> dict:
        """Run a small experiment: evaluate each query and return metrics."""
        if not req.queries:
            raise HTTPException(status_code=422, detail="queries list is empty")

        pipeline = get_pipeline(req.chunking, req.retrieval, req.rerank)
        run_results = []

        for query in req.queries:
            result = pipeline.run(query=query)
            eval_data = evaluate_run(query, result.answer, result.retrieved_docs)
            run_results.append({
                **result.to_dict(),
                **eval_data,
            })

        faithfulness_vals = [r["faithfulness"] for r in run_results]
        relevancy_vals = [r["relevancy"] for r in run_results]

        return {
            "config": {"chunking": req.chunking, "retrieval": req.retrieval, "rerank": req.rerank},
            "num_queries": len(run_results),
            "avg_faithfulness": sum(faithfulness_vals) / len(faithfulness_vals),
            "avg_relevancy": sum(relevancy_vals) / len(relevancy_vals),
            "runs": run_results,
        }
