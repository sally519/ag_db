from __future__ import annotations

from functools import lru_cache

from rag_db.config import Settings
from rag_db.document_loader import DocumentLoader
from rag_db.embedding_client import EmbeddingClient
from rag_db.rag import IngestionPipeline, RetrievalPipeline
from rag_db.services.document_ingestion import DocumentIngestionService
from rag_db.services.ingest_tasks import DocumentIngestTaskService
from rag_db.services.reranker import EmbeddingReranker
from rag_db.services.search import DocumentSearchService
from rag_db.vector_store import ChromaVectorStore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局单例配置对象。"""
    return Settings()


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    """构造并缓存文档入库服务。

    这里统一完成配置、文档加载器、embedding 客户端和向量存储的装配，
    供 FastAPI 依赖注入直接复用。
    """
    settings = get_settings()
    vector_store = ChromaVectorStore(settings.chroma_dir)
    pipeline = IngestionPipeline(vector_store)
    return DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=EmbeddingClient(settings),
        pipeline=pipeline,
    )


@lru_cache(maxsize=1)
def get_document_ingest_task_service() -> DocumentIngestTaskService:
    """构造并缓存异步入库任务服务。"""
    return DocumentIngestTaskService(
        get_document_ingestion_service(),
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_document_search_service() -> DocumentSearchService:
    """构造并缓存查询服务。

    查询服务会复用同一份 embedding 客户端和 Chroma 存储实例，
    以减少重复初始化带来的延迟。
    """
    settings = get_settings()
    vector_store = ChromaVectorStore(settings.chroma_dir)
    retrieval_pipeline = RetrievalPipeline(vector_store)
    embedding_client = EmbeddingClient(settings)
    reranker = EmbeddingReranker(embedding_client)
    return DocumentSearchService(
        settings=settings,
        retrieval_pipeline=retrieval_pipeline,
        embedding_client=embedding_client,
        reranker=reranker,
    )
