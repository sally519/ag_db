from abc import ABC, abstractmethod

from rag_db.models import DocumentChunk, SearchResult


class VectorStore(ABC):
    """Abstract vector store interface for future implementations."""

    @abstractmethod
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

