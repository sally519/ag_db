from rag_db.models import SearchResult
from rag_db.vector_store import VectorStore


class RetrievalPipeline:
    """Coordinates vector search and result handoff to downstream RAG steps."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return self.vector_store.search(query_embedding=query_embedding, top_k=top_k)

