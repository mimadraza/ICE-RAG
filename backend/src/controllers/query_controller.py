from fastapi import HTTPException
from src.models.schemas import QueryRequest, QueryResult
from src.models.pipeline import get_pipeline


class QueryController:
    """Handles all query-related business logic."""

    @staticmethod
    def handle_query(req: QueryRequest) -> dict:
        """Run a single query through the RAG pipeline."""
        _validate_config(req.chunking, req.retrieval)

        try:
            pipeline = get_pipeline(req.chunking, req.retrieval, req.rerank)
            result: QueryResult = pipeline.run(
                query=req.query,
                retrieve_k=req.retrieve_k,
                final_k=req.final_k,
            )
            return result.to_dict()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    @staticmethod
    def handle_default_query(query: str) -> dict:
        """Run query with the best known config (recursive + hybrid + rerank)."""
        req = QueryRequest(
            query=query,
            chunking="recursive",
            retrieval="hybrid",
            rerank=True,
        )
        return QueryController.handle_query(req)


def _validate_config(chunking: str, retrieval: str):
    valid_chunking = {"fixed", "recursive", "semantic"}
    valid_retrieval = {"semantic", "hybrid"}
    if chunking not in valid_chunking:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid chunking '{chunking}'. Choose from: {valid_chunking}",
        )
    if retrieval not in valid_retrieval:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid retrieval '{retrieval}'. Choose from: {valid_retrieval}",
        )
