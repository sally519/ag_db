from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    source: str
    source_type: str
    file_name: str
    media_type: str | None
    text: str


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DuplicateDocumentMatch:
    document_id: str
    chunk_ids: list[str]
    content_hash: str
    source: str | None
    file_name: str | None
    similarity_score: float | None = None


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    collection_name: str
    source: str
    source_type: str
    file_name: str
    chunk_count: int
    embedding_dimension: int
    model_name: str
    content_hash: str
    duplicate_detected: bool = False
    duplicate_strategy: str = "skip"
    status: str = "ingested"
    existing_document_id: str | None = None
    similar_document_detected: bool = False
    similarity_strategy: str = "off"
    similar_document_id: str | None = None
    similarity_score: float | None = None


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestTaskState:
    task_id: str
    status: str
    progress_percent: int
    stage: str
    message: str
    created_at: datetime
    updated_at: datetime
    request: dict[str, Any] = field(default_factory=dict)
    result: IngestionResult | None = None
    error: str | None = None
