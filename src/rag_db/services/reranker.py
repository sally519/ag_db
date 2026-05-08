from __future__ import annotations

from rag_db.embedding_client import EmbeddingClient
from rag_db.models import SearchResult


class EmbeddingReranker:
    """Default local reranker based on query-document embedding similarity."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client

    def rerank(
        self,
        *,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        query_embeddings, _, _ = self.embedding_client.embed_queries([query])
        query_embedding = query_embeddings[0]

        rescored: list[SearchResult] = []
        doc_embeddings = self._resolve_document_embeddings(candidates)
        for candidate, doc_embedding in zip(candidates, doc_embeddings, strict=False):
            rerank_score = _dot(query_embedding, doc_embedding)
            rescored.append(
                SearchResult(
                    chunk_id=candidate.chunk_id,
                    score=rerank_score,
                    content=candidate.content,
                    collection_name=candidate.collection_name,
                    embedding=candidate.embedding,
                    metadata=dict(candidate.metadata),
                    recall_score=candidate.recall_score if candidate.recall_score is not None else candidate.score,
                    rerank_score=rerank_score,
                )
            )

        rescored.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)
        return rescored[:top_n]

    def _resolve_document_embeddings(self, candidates: list[SearchResult]) -> list[list[float]]:
        if all(candidate.embedding is not None for candidate in candidates):
            return [list(candidate.embedding) for candidate in candidates if candidate.embedding is not None]

        doc_embeddings, _, _ = self.embedding_client.embed_documents(
            [candidate.content for candidate in candidates]
        )
        return doc_embeddings


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))
