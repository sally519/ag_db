from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentIngestRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Local file path, file:// path, or URL.")
    collection_name: str | None = Field(default=None, min_length=1)
    document_id: str = Field(default_factory=lambda: uuid4().hex)
    chunk_size: int | None = Field(default=None, ge=1)
    chunk_overlap: int | None = Field(default=None, ge=0)
    duplicate_strategy: Literal["skip", "reject", "replace"] = "skip"
    similarity_threshold: float | None = Field(default=0.98, ge=0.0, le=1.0)
    similarity_strategy: Literal["off", "skip", "reject", "replace"] = "skip"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def validate_source(cls, source: str) -> str:
        normalized = source.strip()
        if not normalized:
            raise ValueError("source must not be empty")
        return normalized


class DocumentIngestResponse(BaseModel):
    success: bool = True
    document_id: str
    collection_name: str
    source: str
    source_type: str
    file_name: str
    chunk_count: int
    embedding_dimension: int
    model_name: str
    content_hash: str
    duplicate_detected: bool
    duplicate_strategy: str
    status: str
    existing_document_id: str | None = None
    similar_document_detected: bool
    similarity_strategy: str
    similar_document_id: str | None = None
    similarity_score: float | None = None


class DocumentIngestTaskCreateResponse(BaseModel):
    success: bool = True
    task_id: str
    status: str
    progress_percent: int
    stage: str
    message: str
    created_at: str
    updated_at: str
    elapsed_seconds: float


class DocumentIngestTaskStatusRequest(BaseModel):
    task_id: str = Field(..., min_length=1)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, task_id: str) -> str:
        normalized = task_id.strip()
        if not normalized:
            raise ValueError("task_id must not be empty")
        return normalized


class DocumentIngestTaskStatusResponse(BaseModel):
    success: bool = True
    task_id: str
    status: str
    progress_percent: int
    stage: str
    message: str
    created_at: str
    updated_at: str
    elapsed_seconds: float
    error: str | None = None
    result: DocumentIngestResponse | None = None
