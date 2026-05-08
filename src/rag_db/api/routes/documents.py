from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_db.api.dependencies import get_document_ingest_task_service
from rag_db.api.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentIngestTaskCreateResponse,
    DocumentIngestTaskStatusRequest,
    DocumentIngestTaskStatusResponse,
)
from rag_db.services.ingest_tasks import DocumentIngestTaskService, IngestTaskNotFoundError


router = APIRouter()


def _elapsed_seconds(task) -> float:
    return round((task.updated_at - task.created_at).total_seconds(), 2)


@router.post("/ingest", response_model=DocumentIngestTaskCreateResponse)
def create_ingest_task(
    request: DocumentIngestRequest,
    service: DocumentIngestTaskService = Depends(get_document_ingest_task_service),
) -> DocumentIngestTaskCreateResponse:
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
