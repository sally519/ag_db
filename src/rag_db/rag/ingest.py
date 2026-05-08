from __future__ import annotations

from rag_db.models import DocumentChunk
from rag_db.vector_store import VectorStore


class IngestionPipeline:
    """封装文档块持久化流程。

    当上游已经完成切块和向量化后，这个类负责把块内容和对应向量写入底层存储。
    """

    def __init__(self, vector_store: VectorStore) -> None:
        """注入底层向量存储实现。"""
        self.vector_store = vector_store

    def ingest(
        self,
        *,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """将文档块和向量一并写入指定集合。"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length must match")
        self.vector_store.upsert(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
        )
