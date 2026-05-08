from rag_db.models import DocumentChunk
from rag_db.vector_store import VectorStore


class IngestionPipeline:
    """Coordinates chunking, embedding, and persistence."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def ingest(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length must match")
        self.vector_store.upsert(chunks, embeddings)

