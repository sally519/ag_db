from __future__ import annotations

from rag_db.models import SearchResult
from rag_db.vector_store import VectorStore


class RetrievalPipeline:
    """封装向量检索流程。

    该类目前保持很薄，主要负责隔离上层服务与底层向量存储之间的直接耦合，
    后续如果增加混合检索、过滤或多阶段召回，可以继续在这里扩展。
    """

    def __init__(self, vector_store: VectorStore) -> None:
        """注入底层向量存储实现。"""
        self.vector_store = vector_store

    def list_searchable_collections(self) -> list[str]:
        """列出当前允许参与查询的集合名称。"""
        return self.vector_store.list_searchable_collections()

    def retrieve(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """在指定集合中执行一次向量召回。"""
        return self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
        )
