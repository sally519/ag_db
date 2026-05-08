from __future__ import annotations

from rag_db.models import SearchResult
from rag_db.vector_store import VectorStore


class RetrievalPipeline:
    """Coordinates vector search and result handoff to downstream RAG steps."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        return self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
        )
