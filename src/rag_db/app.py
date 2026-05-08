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
    """在服务启动阶段预热依赖。

    当前会提前构造文档入库服务和任务服务，从而触发 embedding SDK 的初始化，
    避免第一次真实请求到来时因为模型加载导致长时间阻塞。
    """
    get_document_ingestion_service()
    get_document_ingest_task_service()
    yield


app = FastAPI(
    title="rag_db",
    version="0.1.0",
    description="基于 FastAPI、Chroma 和本地 embedding_engine 的 RAG 向量服务。",
    lifespan=lifespan,
)
app.include_router(documents_router, prefix="/api/documents", tags=["文档"])


@app.post("/health")
def health() -> dict[str, str]:
    """返回服务基础健康状态。

    该接口仅用于确认 FastAPI 进程已经可用，不代表模型、Chroma 或外部依赖
    一定处于完全可查询状态。
    """
    return {"status": "ok"}
