from __future__ import annotations

from abc import ABC, abstractmethod

from rag_db.models import DocumentChunk, DuplicateDocumentMatch, SearchResult


class VectorStore(ABC):
    """向量存储抽象接口。

    该接口定义了当前项目需要的最小能力集合，
    便于后续在 `Chroma`、`FAISS`、`pgvector` 等实现之间切换。
    """

    @abstractmethod
    def list_searchable_collections(self) -> list[str]:
        """列出所有允许参与查询的集合名称。"""
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        *,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """写入或覆盖一批文档块及其向量。"""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """在指定集合中执行向量检索。"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, *, collection_name: str, chunk_ids: list[str]) -> None:
        """按 chunk 编号删除向量记录。"""
        raise NotImplementedError

    @abstractmethod
    def find_duplicate_by_content_hash(
        self,
        *,
        collection_name: str,
        content_hash: str,
    ) -> DuplicateDocumentMatch | None:
        """按内容哈希查找完全重复文档。"""
        raise NotImplementedError

    @abstractmethod
    def upsert_document_record(
        self,
        *,
        collection_name: str,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, object],
    ) -> None:
        """写入文档级记录，用于高相似文档检测。"""
        raise NotImplementedError

    @abstractmethod
    def delete_document_record(self, *, collection_name: str, document_id: str) -> None:
        """删除文档级记录。"""
        raise NotImplementedError

    @abstractmethod
    def find_similar_document(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ) -> DuplicateDocumentMatch | None:
        """查找超过阈值的高相似文档。"""
        raise NotImplementedError
