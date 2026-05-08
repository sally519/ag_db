from __future__ import annotations

import hashlib
import re
from typing import Callable

from rag_db.api.schemas import DocumentIngestRequest
from rag_db.chunking import TextChunker
from rag_db.config import Settings
from rag_db.document_loader import DocumentLoader
from rag_db.embedding_client import EmbeddingClient
from rag_db.models import DocumentChunk, DuplicateDocumentMatch, IngestionResult
from rag_db.rag import IngestionPipeline


class DuplicateDocumentError(ValueError):
    """Raised when a duplicate document is rejected."""


class DocumentIngestionService:
    """End-to-end document ingestion workflow."""

    def __init__(
        self,
        *,
        settings: Settings,
        loader: DocumentLoader,
        embedding_client: EmbeddingClient,
        pipeline: IngestionPipeline,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.embedding_client = embedding_client
        self.pipeline = pipeline

    def ingest(
        self,
        request: DocumentIngestRequest,
        *,
        progress_callback: Callable[[str, int, str], None] | None = None,
    ) -> IngestionResult:
        self._report_progress(progress_callback, "loading_document", 10, "Loading source document")
        source_document = self.loader.load(request.source)
        collection_name = request.collection_name or self.settings.default_collection
        self._report_progress(progress_callback, "checking_duplicate", 25, "Checking duplicates")
        content_hash = _build_content_hash(source_document.text)

        duplicate_match = self.pipeline.vector_store.find_duplicate_by_content_hash(
            collection_name=collection_name,
            content_hash=content_hash,
        )
        if duplicate_match is not None:
            duplicate_result = self._handle_duplicate(
                request=request,
                collection_name=collection_name,
                source_document=source_document,
                duplicate_match=duplicate_match,
                content_hash=content_hash,
            )
            if duplicate_result is not None:
                self._report_progress(
                    progress_callback,
                    "completed",
                    100,
                    f"Duplicate handled with strategy={request.duplicate_strategy}",
                )
                return duplicate_result

        self._report_progress(progress_callback, "document_embedding", 35, "Embedding full document")
        document_embedding, document_dimension, document_model_name = self._embed_full_document(
            source_document.text,
            progress_callback=progress_callback,
        )
        similar_match = self._find_similar_document(
            request=request,
            collection_name=collection_name,
            document_embedding=document_embedding,
        )
        if similar_match is not None:
            similar_result = self._handle_similar_document(
                request=request,
                collection_name=collection_name,
                source_document=source_document,
                similar_match=similar_match,
                content_hash=content_hash,
            )
            if similar_result is not None:
                similar_result.embedding_dimension = document_dimension
                similar_result.model_name = document_model_name
                self._report_progress(
                    progress_callback,
                    "completed",
                    100,
                    f"Similar document handled with strategy={request.similarity_strategy}",
                )
                return similar_result

        chunk_size = request.chunk_size or self.settings.chunk_size
        chunk_overlap = request.chunk_overlap or self.settings.chunk_overlap
        self._report_progress(progress_callback, "chunking", 40, "Splitting document into chunks")
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = chunker.split_text(source_document.text)
        if not texts:
            raise ValueError("document does not contain any non-empty text chunks")

        self._report_progress(progress_callback, "chunk_embedding", 55, "Preparing chunk embeddings")
        embeddings, dimension, model_name = self.embedding_client.embed_documents_with_progress(
            texts,
            progress_callback=lambda batch_index, total_batches: self._report_progress(
                progress_callback,
                "chunk_embedding",
                55 + int(batch_index / total_batches * 35),
                f"Embedding chunks {batch_index}/{total_batches}",
            ),
        )
        chunks = _build_chunks(
            document_id=request.document_id,
            collection_name=collection_name,
            source_document=source_document,
            texts=texts,
            content_hash=content_hash,
            metadata=request.metadata,
        )
        self._report_progress(progress_callback, "storing", 90, "Writing chunks to vector store")
        self.pipeline.ingest(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
        )

        result = IngestionResult(
            document_id=request.document_id,
            collection_name=collection_name,
            source=source_document.source,
            source_type=source_document.source_type,
            file_name=source_document.file_name,
            chunk_count=len(chunks),
            embedding_dimension=dimension,
            model_name=model_name,
            content_hash=content_hash,
            duplicate_detected=duplicate_match is not None,
            duplicate_strategy=request.duplicate_strategy,
            status="replaced"
            if duplicate_match is not None or similar_match is not None
            else "ingested",
            existing_document_id=duplicate_match.document_id if duplicate_match is not None else None,
            similar_document_detected=similar_match is not None,
            similarity_strategy=request.similarity_strategy,
            similar_document_id=similar_match.document_id if similar_match is not None else None,
            similarity_score=similar_match.similarity_score if similar_match is not None else None,
        )
        self.pipeline.vector_store.upsert_document_record(
            collection_name=collection_name,
            document_id=request.document_id,
            embedding=document_embedding,
            metadata={
                "document_id": request.document_id,
                "content_hash": content_hash,
                "source": source_document.source,
                "source_type": source_document.source_type,
                "file_name": source_document.file_name,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
            },
        )
        self._report_progress(progress_callback, "completed", 100, "Document ingested")
        return result

    def _handle_duplicate(
        self,
        *,
        request: DocumentIngestRequest,
        collection_name: str,
        source_document,
        duplicate_match: DuplicateDocumentMatch,
        content_hash: str,
    ) -> IngestionResult | None:
        if request.duplicate_strategy == "reject":
            raise DuplicateDocumentError(
                f"duplicate document detected: existing document_id={duplicate_match.document_id}"
            )
        if request.duplicate_strategy == "skip":
            return IngestionResult(
                document_id=duplicate_match.document_id,
                collection_name=collection_name,
                source=source_document.source,
                source_type=source_document.source_type,
                file_name=source_document.file_name,
                chunk_count=0,
                embedding_dimension=0,
                model_name="",
                content_hash=content_hash,
                duplicate_detected=True,
                duplicate_strategy=request.duplicate_strategy,
                status="skipped",
                existing_document_id=duplicate_match.document_id,
            )
        self.pipeline.vector_store.delete(
            collection_name=collection_name,
            chunk_ids=duplicate_match.chunk_ids,
        )
        self.pipeline.vector_store.delete_document_record(
            collection_name=collection_name,
            document_id=duplicate_match.document_id,
        )
        return None

    def _find_similar_document(
        self,
        *,
        request: DocumentIngestRequest,
        collection_name: str,
        document_embedding: list[float],
    ) -> DuplicateDocumentMatch | None:
        if request.similarity_strategy == "off" or request.similarity_threshold is None:
            return None
        return self.pipeline.vector_store.find_similar_document(
            collection_name=collection_name,
            query_embedding=document_embedding,
            similarity_threshold=request.similarity_threshold,
        )

    def _handle_similar_document(
        self,
        *,
        request: DocumentIngestRequest,
        collection_name: str,
        source_document,
        similar_match: DuplicateDocumentMatch,
        content_hash: str,
    ) -> IngestionResult | None:
        if request.similarity_strategy == "reject":
            raise DuplicateDocumentError(
                "similar document detected: "
                f"existing document_id={similar_match.document_id}, "
                f"similarity={similar_match.similarity_score:.4f}"
            )
        if request.similarity_strategy == "skip":
            return IngestionResult(
                document_id=similar_match.document_id,
                collection_name=collection_name,
                source=source_document.source,
                source_type=source_document.source_type,
                file_name=source_document.file_name,
                chunk_count=0,
                embedding_dimension=0,
                model_name="",
                content_hash=content_hash,
                duplicate_detected=False,
                duplicate_strategy=request.duplicate_strategy,
                status="skipped_similar",
                existing_document_id=None,
                similar_document_detected=True,
                similarity_strategy=request.similarity_strategy,
                similar_document_id=similar_match.document_id,
                similarity_score=similar_match.similarity_score,
            )
        self.pipeline.vector_store.delete(
            collection_name=collection_name,
            chunk_ids=similar_match.chunk_ids,
        )
        self.pipeline.vector_store.delete_document_record(
            collection_name=collection_name,
            document_id=similar_match.document_id,
        )
        return None

    def _embed_full_document(
        self,
        text: str,
        *,
        progress_callback: Callable[[str, int, str], None] | None = None,
    ) -> tuple[list[float], int, str]:
        embeddings, dimension, model_name = self.embedding_client.embed_documents_with_progress(
            [text],
            progress_callback=lambda batch_index, total_batches: self._report_progress(
                progress_callback,
                "document_embedding",
                35 + int(batch_index / total_batches * 10),
                f"Embedding full document {batch_index}/{total_batches}",
            ),
        )
        return embeddings[0], dimension, model_name

    @staticmethod
    def _report_progress(
        progress_callback: Callable[[str, int, str], None] | None,
        stage: str,
        progress_percent: int,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(stage, progress_percent, message)


def _build_chunks(
    *,
    document_id: str,
    collection_name: str,
    source_document,
    texts: list[str],
    content_hash: str,
    metadata: dict[str, object],
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    total_chunks = len(texts)
    for index, text in enumerate(texts):
        chunk_id = f"{document_id}:{index}"
        chunk_metadata = {
            "document_id": document_id,
            "collection_name": collection_name,
            "chunk_index": index,
            "chunk_count": total_chunks,
            "source": source_document.source,
            "source_type": source_document.source_type,
            "file_name": source_document.file_name,
            "content_hash": content_hash,
            **metadata,
        }
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=text,
                metadata=chunk_metadata,
            )
        )
    return chunks


def _build_content_hash(text: str) -> str:
    normalized = _normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_text_for_hash(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\s+", " ", cleaned)
