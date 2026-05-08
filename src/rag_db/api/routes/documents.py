from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_db.api.dependencies import (
    get_document_ingest_task_service,
    get_document_search_service,
)
from rag_db.api.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentIngestTaskCreateResponse,
    DocumentIngestTaskStatusRequest,
    DocumentIngestTaskStatusResponse,
    SearchHitResponse,
)
from rag_db.services.ingest_tasks import DocumentIngestTaskService, IngestTaskNotFoundError
from rag_db.services.search import DocumentSearchService, SearchConcurrencyLimitError


router = APIRouter()


def _elapsed_seconds(task) -> float:
    """根据任务创建时间和更新时间计算已耗时秒数。"""
    return round((task.updated_at - task.created_at).total_seconds(), 2)


@router.post("/ingest", response_model=DocumentIngestTaskCreateResponse)
def create_ingest_task(
    request: DocumentIngestRequest,
    service: DocumentIngestTaskService = Depends(get_document_ingest_task_service),
) -> DocumentIngestTaskCreateResponse:
    """创建文档入库后台任务。

    该接口只负责登记任务并立即返回任务编号，
    真正的文档读取、切块、向量化和入库逻辑在后台线程中执行。
    """
    task = service.create_task(request)
    return DocumentIngestTaskCreateResponse(
        task_id=task.task_id,
        status=task.status,
        progress_percent=task.progress_percent,
        stage=task.stage,
        message=task.message,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        elapsed_seconds=_elapsed_seconds(task),
    )


@router.post("/ingest/status", response_model=DocumentIngestTaskStatusResponse)
def get_ingest_task_status(
    request: DocumentIngestTaskStatusRequest,
    service: DocumentIngestTaskService = Depends(get_document_ingest_task_service),
) -> DocumentIngestTaskStatusResponse:
    """查询后台入库任务当前状态。"""
    try:
        task = service.get_task(request.task_id)
    except IngestTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"task not found: {request.task_id}") from exc

    result = None
    if task.result is not None:
        result = DocumentIngestResponse(
            document_id=task.result.document_id,
            collection_name=task.result.collection_name,
            source=task.result.source,
            source_type=task.result.source_type,
            file_name=task.result.file_name,
            chunk_count=task.result.chunk_count,
            embedding_dimension=task.result.embedding_dimension,
            model_name=task.result.model_name,
            content_hash=task.result.content_hash,
            duplicate_detected=task.result.duplicate_detected,
            duplicate_strategy=task.result.duplicate_strategy,
            status=task.result.status,
            existing_document_id=task.result.existing_document_id,
            similar_document_detected=task.result.similar_document_detected,
            similarity_strategy=task.result.similarity_strategy,
            similar_document_id=task.result.similar_document_id,
            similarity_score=task.result.similarity_score,
        )

    return DocumentIngestTaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress_percent=task.progress_percent,
        stage=task.stage,
        message=task.message,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        elapsed_seconds=_elapsed_seconds(task),
        error=task.error,
        result=result,
    )


@router.post("/search", response_model=DocumentSearchResponse)
def search_documents(
    request: DocumentSearchRequest,
    service: DocumentSearchService = Depends(get_document_search_service),
) -> DocumentSearchResponse:
    """执行文档查询。

    当前查询逻辑会在所有可搜索集合中做召回，
    然后对召回候选统一重排，最后返回重排后的结果。
    """
    try:
        hits = service.search(
            query=request.query,
            recall_top_k=request.recall_top_k,
            rerank_top_n=request.rerank_top_n,
        )
    except SearchConcurrencyLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    searched_collections = sorted(
        {
            hit.collection_name
            for hit in hits
            if hit.collection_name
        }
    )
    return DocumentSearchResponse(
        query=request.query,
        searched_collections=searched_collections,
        recall_top_k=request.recall_top_k,
        rerank_top_n=request.rerank_top_n,
        hits=[
            SearchHitResponse(
                chunk_id=hit.chunk_id,
                collection_name=hit.collection_name or "",
                content=hit.content,
                metadata=hit.metadata,
                recall_score=hit.recall_score or 0.0,
                rerank_score=hit.rerank_score or hit.score,
            )
            for hit in hits
        ],
    )
