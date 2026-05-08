from __future__ import annotations

from rag_db.embedding_client import EmbeddingClient
from rag_db.models import SearchResult


class EmbeddingReranker:
    """基于向量相似度的本地重排器。

    当前实现不依赖单独的 cross-encoder 模型，而是使用：
    - 查询向量
    - 候选文档向量

    通过向量点积对候选结果再次排序。这样实现简单、依赖少，
    也便于在本地环境先跑通完整查询链路。
    """

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        """注入 embedding 客户端。"""
        self.embedding_client = embedding_client

    def rerank(
        self,
        *,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        """对召回候选进行重排，并只返回前 `top_n` 条。"""
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
        """优先复用召回阶段已拿到的向量，必要时才重新向量化文本。"""
        if all(candidate.embedding is not None for candidate in candidates):
            return [list(candidate.embedding) for candidate in candidates if candidate.embedding is not None]

        doc_embeddings, _, _ = self.embedding_client.embed_documents(
            [candidate.content for candidate in candidates]
        )
        return doc_embeddings


def _dot(left: list[float], right: list[float]) -> float:
    """计算两个向量的点积。"""
    return float(sum(a * b for a, b in zip(left, right, strict=False)))
