from fastapi import APIRouter
from src.models.schemas import QueryRequest, ExperimentRequest
from src.controllers.query_controller import QueryController
from src.controllers.experiment_controller import ExperimentController
from src.controllers.info_controller import InfoController
from src.views.formatters import (
    query_view,
    experiment_summary_view,
    experiment_run_view,
    info_view,
)

router = APIRouter()


# ── Info & Health ────────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
def health():
    return InfoController.health()


@router.get("/info", tags=["system"])
def info():
    return info_view(InfoController.info())


# ── Query ────────────────────────────────────────────────────────────────────────

@router.post("/query", tags=["query"])
def query(req: QueryRequest):
    """Run a single RAG query with the specified config."""
    result = QueryController.handle_query(req)
    return query_view(result)


@router.get("/query/default", tags=["query"])
def query_default(q: str):
    """Shortcut: run a query with the best known config (recursive + hybrid + rerank)."""
    result = QueryController.handle_default_query(q)
    return query_view(result)


# ── Experiments ──────────────────────────────────────────────────────────────────

@router.get("/experiments/summary", tags=["experiments"])
def experiments_summary():
    """Return the cached experiment summary results."""
    data = ExperimentController.get_summary()
    return experiment_summary_view(data)


@router.get("/experiments/chunking-report", tags=["experiments"])
def chunking_report():
    """Return the chunking report text."""
    return ExperimentController.get_chunking_report()


@router.post("/experiments/run", tags=["experiments"])
def run_experiment(req: ExperimentRequest):
    """Run a fresh experiment with evaluation (faithfulness + relevancy)."""
    data = ExperimentController.run_experiment(req)
    return experiment_run_view(data)
