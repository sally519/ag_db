from __future__ import annotations

from datetime import datetime
from threading import Lock, Thread
from uuid import uuid4

from rag_db.api.schemas import DocumentIngestRequest
from rag_db.models import IngestTaskState, IngestionResult
from rag_db.services.document_ingestion import DocumentIngestionService


class IngestTaskNotFoundError(KeyError):
    """Raised when a task id does not exist."""


class DocumentIngestTaskService:
    """In-memory background task manager for document ingestion."""

    def __init__(self, ingestion_service: DocumentIngestionService) -> None:
        self.ingestion_service = ingestion_service
        self._tasks: dict[str, IngestTaskState] = {}
        self._lock = Lock()

    def create_task(self, request: DocumentIngestRequest) -> IngestTaskState:
        task_id = uuid4().hex
        task = IngestTaskState(
            task_id=task_id,
            status="pending",
            progress_percent=0,
            stage="queued",
            message="Task queued",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            request=request.model_dump(mode="json"),
        )
        with self._lock:
            self._tasks[task_id] = task

        worker = Thread(
            target=self._run_task,
            args=(task_id, request),
            daemon=True,
            name=f"document-ingest-{task_id[:8]}",
        )
        worker.start()
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> IngestTaskState:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise IngestTaskNotFoundError(task_id)
            return IngestTaskState(
                task_id=task.task_id,
                status=task.status,
                progress_percent=task.progress_percent,
                stage=task.stage,
                message=task.message,
                created_at=task.created_at,
                updated_at=task.updated_at,
                request=dict(task.request),
                result=task.result,
                error=task.error,
            )

    def _run_task(self, task_id: str, request: DocumentIngestRequest) -> None:
        self._update_task(
            task_id,
            status="running",
            progress_percent=5,
            stage="starting",
            message="Task started",
        )
        try:
            result = self.ingestion_service.ingest(
                request,
                progress_callback=lambda stage, percent, message: self._update_task(
                    task_id,
                    status="running",
                    progress_percent=percent,
                    stage=stage,
                    message=message,
                ),
            )
            self._complete_task(task_id, result)
        except Exception as exc:
            self._fail_task(task_id, str(exc))

    def _complete_task(self, task_id: str, result: IngestionResult) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "completed"
            task.progress_percent = 100
            task.stage = "completed"
            task.message = "Task completed"
            task.updated_at = datetime.now()
            task.result = result
            task.error = None

    def _fail_task(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "failed"
            task.progress_percent = task.progress_percent or 100
            task.stage = "failed"
            task.message = "Task failed"
            task.updated_at = datetime.now()
            task.error = error

    def _update_task(
        self,
        task_id: str,
        *,
        status: str,
        progress_percent: int,
        stage: str,
        message: str,
    ) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = status
            task.progress_percent = progress_percent
            task.stage = stage
            task.message = message
            task.updated_at = datetime.now()
