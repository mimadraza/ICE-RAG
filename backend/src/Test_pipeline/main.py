from fastapi import FastAPI
from src.module.api.router import router

app = FastAPI(
      title="RAG Experiment API",
      description="Endpoints to launch experiments and pull out faithfulness/relevance.",
      version="1.0.0",
  )

  # include the MVC router
app.include_router(router, prefix="/api")


@router.get("/auto/run-recursive-hybrid")
def auto_run():
      spec = {
          "queries": [
              {"query": "Your query here"}
          ]
      }
      return _run_experiment(spec)