from __future__ import annotations

from pathlib import Path
import json

from rag_db.models import DocumentChunk, DuplicateDocumentMatch, SearchResult
from rag_db.vector_store.base import VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma-backed persistent vector store."""

    def __init__(self, persist_directory: Path) -> None:
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
        collection = self._get_collection(collection_name)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_normalize_metadata(chunk.metadata) for chunk in chunks],
        )

    def search(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        collection = self._get_collection(collection_name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            SearchResult(
                chunk_id=chunk_id,
                score=distance,
                content=document,
                metadata=metadata or {},
            )
            for chunk_id, document, distance, metadata in zip(
                ids,
                documents,
                distances,
                metadatas,
                strict=False,
            )
        ]

    def delete(self, *, collection_name: str, chunk_ids: list[str]) -> None:
        collection = self._get_collection(collection_name)
        collection.delete(ids=chunk_ids)

    def find_duplicate_by_content_hash(
        self,
        *,
        collection_name: str,
        content_hash: str,
    ) -> DuplicateDocumentMatch | None:
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
        collection = self._get_document_collection(collection_name)
        collection.upsert(
            ids=[document_id],
            documents=[str(metadata.get("file_name", document_id))],
            embeddings=[embedding],
            metadatas=[_normalize_metadata(metadata)],
        )

    def delete_document_record(self, *, collection_name: str, document_id: str) -> None:
        collection = self._get_document_collection(collection_name)
        collection.delete(ids=[document_id])

    def find_similar_document(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ) -> DuplicateDocumentMatch | None:
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
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_document_collection(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=f"{collection_name}__documents",
            metadata={"hnsw:space": "cosine"},
        )


def _normalize_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
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
    if value in (None, ""):
        return None
    return str(value)


def _parse_chunk_ids(value: object) -> list[str]:
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
