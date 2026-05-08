from __future__ import annotations

from rag_db.embedding_client import EmbeddingClient
from rag_db.models import SearchResult
from rag_db.rag import RetrievalPipeline
from rag_db.services.reranker import EmbeddingReranker


class DocumentSearchService:
    """文档查询服务。

    当前查询流程分为两段：
    1. 在所有可搜索集合中做向量召回
    2. 对合并后的候选统一做重排

    这样既能隐藏底层集合细节，也能为后续多知识库检索保留扩展空间。
    """

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
        embedding_client: EmbeddingClient,
        reranker: EmbeddingReranker,
    ) -> None:
        """注入召回、向量化和重排依赖。"""
        self.retrieval_pipeline = retrieval_pipeline
        self.embedding_client = embedding_client
        self.reranker = reranker

    def search(
        self,
        *,
        query: str,
        recall_top_k: int = 10,
        rerank_top_n: int = 3,
    ) -> list[SearchResult]:
        """执行一次完整查询。

        注意当前 `recall_top_k` 的语义是全局召回上限：
        每个集合先取自己的候选，最终会汇总后再裁剪为全局前 `recall_top_k` 条，
        然后进入重排阶段。
        """
        query_embeddings, _, _ = self.embedding_client.embed_queries([query])
        collections = self.retrieval_pipeline.list_searchable_collections()
        normalized: list[SearchResult] = []
        for collection_name in collections:
            recalled = self.retrieval_pipeline.retrieve(
                collection_name=collection_name,
                query_embedding=query_embeddings[0],
                top_k=recall_top_k,
            )
            normalized.extend(
                SearchResult(
                    chunk_id=item.chunk_id,
                    score=_distance_to_similarity(item.score),
                    content=item.content,
                    collection_name=collection_name,
                    embedding=item.embedding,
                    metadata=dict(item.metadata),
                    recall_score=_distance_to_similarity(item.score),
                )
                for item in recalled
            )
        normalized.sort(key=lambda item: item.recall_score or 0.0, reverse=True)
        normalized = normalized[:recall_top_k]
        return self.reranker.rerank(query=query, candidates=normalized, top_n=rerank_top_n)


def _distance_to_similarity(distance: float) -> float:
    """将距离分数转换为越大越好的相似度分数。"""
    return max(0.0, 1.0 - float(distance))
