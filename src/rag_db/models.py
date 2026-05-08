from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    """统一表示加载后的源文档。

    该对象承载文档原始来源信息以及已经抽取好的纯文本，
    是后续切块、去重和向量化流程的基础输入。
    """
    source: str
    source_type: str
    file_name: str
    media_type: str | None
    text: str


@dataclass(slots=True)
class DocumentChunk:
    """表示单个文档块及其附属元数据。"""
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DuplicateDocumentMatch:
    """表示重复文档或高相似文档命中结果。"""
    document_id: str
    chunk_ids: list[str]
    content_hash: str
    source: str | None
    file_name: str | None
    similarity_score: float | None = None


@dataclass(slots=True)
class IngestionResult:
    """表示一次文档入库流程的最终结果。"""
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
    """表示一次检索或重排后的候选结果。

    `score` 字段在不同阶段含义不同：
    - 召回阶段通常表示原始距离或转换后的相似度
    - 重排阶段通常表示最终排序分数
    """
    chunk_id: str
    score: float
    content: str
    collection_name: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recall_score: float | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class IngestTaskState:
    """表示异步入库任务在某一时刻的状态快照。"""
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
