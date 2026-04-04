from dotenv import load_dotenv
from pathlib import Path
import os
import logging
logging.basicConfig(level=logging.INFO)
load_dotenv(Path(__file__).parent / ".env")
_key = os.getenv("GROQ_API_KEY", "")
logging.getLogger(__name__).info("GROQ_API_KEY present: %s, length: %d", bool(_key), len(_key))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.api import router

app = FastAPI(
    title="RAG Experiment API",
    description=(
        "REST API for a Retrieval-Augmented Generation (RAG) pipeline. "
        "Supports recursive/fixed/semantic chunking, hybrid+semantic retrieval, "
        "cross-encoder reranking, and LLM-based evaluation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", tags=["root"])
def root():
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/api/health",
    }
