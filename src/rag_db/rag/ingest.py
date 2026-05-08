from __future__ import annotations

from rag_db.models import DocumentChunk
from rag_db.vector_store import VectorStore


class IngestionPipeline:
    """Coordinates persistence after chunking and embedding are complete."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def ingest(
        self,
        *,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length must match")
        self.vector_store.upsert(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
        )
