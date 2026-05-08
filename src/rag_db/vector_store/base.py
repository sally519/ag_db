from __future__ import annotations

from abc import ABC, abstractmethod

from rag_db.models import DocumentChunk, DuplicateDocumentMatch, SearchResult


class VectorStore(ABC):
    """Abstract vector store interface for future implementations."""

    @abstractmethod
    def upsert(
        self,
        *,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, *, collection_name: str, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def find_duplicate_by_content_hash(
        self,
        *,
        collection_name: str,
        content_hash: str,
    ) -> DuplicateDocumentMatch | None:
        raise NotImplementedError

    @abstractmethod
    def upsert_document_record(
        self,
        *,
        collection_name: str,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, object],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_document_record(self, *, collection_name: str, document_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def find_similar_document(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ) -> DuplicateDocumentMatch | None:
        raise NotImplementedError
