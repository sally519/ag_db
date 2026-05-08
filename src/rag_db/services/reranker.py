from __future__ import annotations

from rag_db.embedding_client import EmbeddingClient
from rag_db.models import SearchResult


class EmbeddingReranker:
    """基于 `embedding_engine` reranker SDK 的本地重排器。

    当前实现直接复用兄弟项目新增的 `create_reranker_sdk` 能力，
    由专门的重排模型对召回候选做语义相关性排序。
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

        reranked_items, _ = self.embedding_client.rerank_documents(
            query=query,
            documents=[candidate.content for candidate in candidates],
            top_n=top_n,
        )

        rescored: list[SearchResult] = []
        for item in reranked_items:
            candidate = candidates[int(item["index"])]
            rerank_score = float(item["relevance_score"])
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
