from __future__ import annotations

from pathlib import Path
import json

from rag_db.models import DocumentChunk, DuplicateDocumentMatch, SearchResult
from rag_db.vector_store.base import VectorStore


class ChromaVectorStore(VectorStore):
    """基于 Chroma 的本地持久化向量存储实现。"""

    def __init__(self, persist_directory: Path) -> None:
        """初始化 Chroma 客户端，并确保持久化目录存在。"""
        from chromadb import PersistentClient

        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(path=str(self.persist_directory))

    def upsert(
        self,
        *,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """批量写入文档块。"""
        collection = self._get_collection(collection_name)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_normalize_metadata(chunk.metadata) for chunk in chunks],
        )

    def list_searchable_collections(self) -> list[str]:
        """返回所有普通检索集合，自动排除文档级辅助集合。"""
        collections = self.client.list_collections()
        names: list[str] = []
        for item in collections:
            name = item if isinstance(item, str) else getattr(item, "name", "")
            if not name or name.endswith("__documents"):
                continue
            names.append(str(name))
        return sorted(set(names))

    def search(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """在指定集合内执行相似度查询，并带回候选文本与向量。"""
        collection = self._get_collection(collection_name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0]
        return [
            SearchResult(
                chunk_id=chunk_id,
                score=distance,
                content=document,
                collection_name=collection_name,
                embedding=embedding,
                metadata=metadata or {},
            )
            for chunk_id, document, distance, metadata, embedding in zip(
                ids,
                documents,
                distances,
                metadatas,
                embeddings,
                strict=False,
            )
        ]

    def delete(self, *, collection_name: str, chunk_ids: list[str]) -> None:
        """删除指定 chunk 记录。"""
        collection = self._get_collection(collection_name)
        collection.delete(ids=chunk_ids)

    def find_duplicate_by_content_hash(
        self,
        *,
        collection_name: str,
        content_hash: str,
    ) -> DuplicateDocumentMatch | None:
        """按内容哈希查找重复文档。"""
        collection = self._get_collection(collection_name)
        result = collection.get(
            where={"content_hash": content_hash},
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        if not ids:
            return None

        first_metadata = (metadatas[0] if metadatas else None) or {}
        return DuplicateDocumentMatch(
            document_id=str(first_metadata.get("document_id", "")),
            chunk_ids=[str(chunk_id) for chunk_id in ids],
            content_hash=content_hash,
            source=_optional_string(first_metadata.get("source")),
            file_name=_optional_string(first_metadata.get("file_name")),
        )

    def upsert_document_record(
        self,
        *,
        collection_name: str,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, object],
    ) -> None:
        """写入文档级向量记录。"""
        collection = self._get_document_collection(collection_name)
        collection.upsert(
            ids=[document_id],
            documents=[str(metadata.get("file_name", document_id))],
            embeddings=[embedding],
            metadatas=[_normalize_metadata(metadata)],
        )

    def delete_document_record(self, *, collection_name: str, document_id: str) -> None:
        """删除文档级向量记录。"""
        collection = self._get_document_collection(collection_name)
        collection.delete(ids=[document_id])

    def find_similar_document(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ) -> DuplicateDocumentMatch | None:
        """查找与当前文档最相近且超过阈值的文档级记录。"""
        collection = self._get_document_collection(collection_name)
        if collection.count() == 0:
            return None
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        if not ids:
            return None

        document_id = str(ids[0])
        distance = float(distances[0])
        similarity_score = max(0.0, 1.0 - distance)
        if similarity_score < similarity_threshold:
            return None

        metadata = (metadatas[0] if metadatas else None) or {}
        chunk_ids = _parse_chunk_ids(metadata.get("chunk_ids"))
        return DuplicateDocumentMatch(
            document_id=document_id,
            chunk_ids=chunk_ids,
            content_hash=str(metadata.get("content_hash", "")),
            source=_optional_string(metadata.get("source")),
            file_name=_optional_string(metadata.get("file_name")),
            similarity_score=similarity_score,
        )

    def _get_collection(self, collection_name: str):
        """获取普通 chunk 集合，不存在时自动创建。"""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_document_collection(self, collection_name: str):
        """获取文档级辅助集合，不存在时自动创建。"""
        return self.client.get_or_create_collection(
            name=f"{collection_name}__documents",
            metadata={"hnsw:space": "cosine"},
        )


def _normalize_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
    """将 metadata 转换为 Chroma 支持的基础类型。"""
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float, str)):
            normalized[key] = value
        elif value is None:
            normalized[key] = ""
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized


def _optional_string(value: object) -> str | None:
    """将空值统一转为 `None`，其余值转成字符串。"""
    if value in (None, ""):
        return None
    return str(value)


def _parse_chunk_ids(value: object) -> list[str]:
    """将 metadata 中存储的 chunk 编号解析为字符串列表。"""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
