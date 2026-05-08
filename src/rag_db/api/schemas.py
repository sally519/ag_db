from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentIngestRequest(BaseModel):
    """文档入库请求体。"""

    source: str = Field(..., min_length=1, description="文档来源，可以是本地路径、file:// 路径或 URL。")
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
        """清理并校验文档来源字符串。"""
        normalized = source.strip()
        if not normalized:
            raise ValueError("source must not be empty")
        return normalized


class DocumentIngestResponse(BaseModel):
    """文档入库最终结果。"""

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
    """创建异步入库任务后的即时响应。"""

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
    """查询异步入库任务状态的请求体。"""

    task_id: str = Field(..., min_length=1)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, task_id: str) -> str:
        """校验任务编号不能为空。"""
        normalized = task_id.strip()
        if not normalized:
            raise ValueError("task_id must not be empty")
        return normalized


class DocumentIngestTaskStatusResponse(BaseModel):
    """异步入库任务状态响应。"""

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


class DocumentSearchRequest(BaseModel):
    """文档查询请求体。"""

    query: str = Field(..., min_length=1)
    recall_top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_n: int = Field(default=3, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """确保用户查询内容不是空白字符串。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class SearchHitResponse(BaseModel):
    """单条检索命中结果。"""

    chunk_id: str
    collection_name: str
    content: str
    metadata: dict[str, Any]
    recall_score: float
    rerank_score: float


class DocumentSearchResponse(BaseModel):
    """查询接口返回结果。"""

    success: bool = True
    query: str
    searched_collections: list[str]
    recall_top_k: int
    rerank_top_n: int
    hits: list[SearchHitResponse]
