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
    return Settings()


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
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
    return DocumentIngestTaskService(get_document_ingestion_service())


@lru_cache(maxsize=1)
def get_document_search_service() -> DocumentSearchService:
    settings = get_settings()
    vector_store = ChromaVectorStore(settings.chroma_dir)
    retrieval_pipeline = RetrievalPipeline(vector_store)
    embedding_client = EmbeddingClient(settings)
    reranker = EmbeddingReranker(embedding_client)
    return DocumentSearchService(
        retrieval_pipeline=retrieval_pipeline,
        embedding_client=embedding_client,
        reranker=reranker,
    )
