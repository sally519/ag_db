from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_db.api.dependencies import (
    get_document_ingest_task_service,
    get_document_ingestion_service,
)
from rag_db.api.routes.documents import router as documents_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_document_ingestion_service()
    get_document_ingest_task_service()
    yield


app = FastAPI(
    title="rag_db",
    version="0.1.0",
    description="Local RAG vector storage service built with FastAPI and Chroma.",
    lifespan=lifespan,
)
app.include_router(documents_router, prefix="/api/documents", tags=["documents"])


@app.post("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
