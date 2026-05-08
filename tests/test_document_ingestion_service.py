from pathlib import Path

from rag_db.api.schemas import DocumentIngestRequest
from rag_db.config import Settings
from rag_db.document_loader import DocumentLoader
from rag_db.embedding_client import EmbeddingClient
from rag_db.rag import IngestionPipeline
from rag_db.services.document_ingestion import (
    DocumentIngestionService,
    DuplicateDocumentError,
)
from rag_db.vector_store.base import VectorStore


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.collection_name = ""
        self.chunks = []
        self.embeddings = []
        self.deleted_chunk_ids = []
        self.duplicate_match = None
        self.similar_match = None
        self.document_records = []

    def upsert(self, *, collection_name: str, chunks, embeddings) -> None:
        self.collection_name = collection_name
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, *, collection_name: str, query_embedding: list[float], top_k: int = 5):
        return []

    def delete(self, *, collection_name: str, chunk_ids: list[str]) -> None:
        self.deleted_chunk_ids = chunk_ids
        return None

    def find_duplicate_by_content_hash(
        self,
        *,
        collection_name: str,
        content_hash: str,
    ):
        return self.duplicate_match

    def upsert_document_record(
        self,
        *,
        collection_name: str,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, object],
    ) -> None:
        self.document_records.append(
            {
                "collection_name": collection_name,
                "document_id": document_id,
                "embedding": embedding,
                "metadata": metadata,
            }
        )

    def delete_document_record(self, *, collection_name: str, document_id: str) -> None:
        self.document_records.append(
            {
                "collection_name": collection_name,
                "document_id": document_id,
                "deleted": True,
            }
        )

    def find_similar_document(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ):
        return self.similar_match


class FakeEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], int, str]:
        return [[0.1, 0.2] for _ in texts], 2, "fake-model"


def test_document_ingestion_service_ingests_local_text(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("one two three four five six seven eight nine ten", encoding="utf-8")

    settings = Settings(
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        chunk_size=12,
        chunk_overlap=3,
    )
    store = FakeVectorStore()
    service = DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=FakeEmbeddingClient(),
        pipeline=IngestionPipeline(store),
    )

    result = service.ingest(
        DocumentIngestRequest(
            source=str(source),
            collection_name="docs",
            document_id="doc-1",
            metadata={"tenant": "demo"},
        )
    )

    assert result.document_id == "doc-1"
    assert result.collection_name == "docs"
    assert result.chunk_count == len(store.chunks)
    assert store.collection_name == "docs"
    assert store.embeddings
    assert store.chunks[0].metadata["tenant"] == "demo"
    assert store.chunks[0].metadata["content_hash"]
    assert result.status == "ingested"
    assert result.similar_document_detected is False
    assert store.document_records


def test_document_ingestion_service_skips_exact_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("same content", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    store = FakeVectorStore()
    store.duplicate_match = type(
        "Match",
        (),
        {
            "document_id": "existing-doc",
            "chunk_ids": ["existing-doc:0"],
            "content_hash": "hash-1",
            "source": "old.txt",
            "file_name": "old.txt",
        },
    )()
    service = DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=FakeEmbeddingClient(),
        pipeline=IngestionPipeline(store),
    )

    result = service.ingest(
        DocumentIngestRequest(
            source=str(source),
            collection_name="docs",
            document_id="doc-2",
            duplicate_strategy="skip",
        )
    )

    assert result.status == "skipped"
    assert result.duplicate_detected is True
    assert result.existing_document_id == "existing-doc"
    assert store.chunks == []


def test_document_ingestion_service_rejects_exact_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("same content", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    store = FakeVectorStore()
    store.duplicate_match = type(
        "Match",
        (),
        {
            "document_id": "existing-doc",
            "chunk_ids": ["existing-doc:0"],
            "content_hash": "hash-1",
            "source": "old.txt",
            "file_name": "old.txt",
        },
    )()
    service = DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=FakeEmbeddingClient(),
        pipeline=IngestionPipeline(store),
    )

    try:
        service.ingest(
            DocumentIngestRequest(
                source=str(source),
                collection_name="docs",
                document_id="doc-3",
                duplicate_strategy="reject",
            )
        )
        assert False, "expected DuplicateDocumentError"
    except DuplicateDocumentError:
        pass


def test_document_ingestion_service_replaces_exact_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("same content", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    store = FakeVectorStore()
    store.duplicate_match = type(
        "Match",
        (),
        {
            "document_id": "existing-doc",
            "chunk_ids": ["existing-doc:0", "existing-doc:1"],
            "content_hash": "hash-1",
            "source": "old.txt",
            "file_name": "old.txt",
        },
    )()
    service = DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=FakeEmbeddingClient(),
        pipeline=IngestionPipeline(store),
    )

    result = service.ingest(
        DocumentIngestRequest(
            source=str(source),
            collection_name="docs",
            document_id="doc-4",
            duplicate_strategy="replace",
        )
    )

    assert store.deleted_chunk_ids == ["existing-doc:0", "existing-doc:1"]
    assert result.status == "replaced"
    assert result.existing_document_id == "existing-doc"


def test_document_ingestion_service_skips_similar_document(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("near duplicate content", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    store = FakeVectorStore()
    store.similar_match = type(
        "Match",
        (),
        {
            "document_id": "similar-doc",
            "chunk_ids": ["similar-doc:0"],
            "content_hash": "hash-2",
            "source": "other.txt",
            "file_name": "other.txt",
            "similarity_score": 0.992,
        },
    )()
    service = DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=FakeEmbeddingClient(),
        pipeline=IngestionPipeline(store),
    )

    result = service.ingest(
        DocumentIngestRequest(
            source=str(source),
            collection_name="docs",
            document_id="doc-5",
            similarity_strategy="skip",
            similarity_threshold=0.98,
        )
    )

    assert result.status == "skipped_similar"
    assert result.similar_document_detected is True
    assert result.similar_document_id == "similar-doc"
    assert result.similarity_score == 0.992


def test_document_ingestion_service_replaces_similar_document(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("near duplicate content", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    store = FakeVectorStore()
    store.similar_match = type(
        "Match",
        (),
        {
            "document_id": "similar-doc",
            "chunk_ids": ["similar-doc:0", "similar-doc:1"],
            "content_hash": "hash-2",
            "source": "other.txt",
            "file_name": "other.txt",
            "similarity_score": 0.992,
        },
    )()
    service = DocumentIngestionService(
        settings=settings,
        loader=DocumentLoader(),
        embedding_client=FakeEmbeddingClient(),
        pipeline=IngestionPipeline(store),
    )

    result = service.ingest(
        DocumentIngestRequest(
            source=str(source),
            collection_name="docs",
            document_id="doc-6",
            similarity_strategy="replace",
            similarity_threshold=0.98,
        )
    )

    assert result.status == "replaced"
    assert result.similar_document_detected is True
    assert result.similar_document_id == "similar-doc"
    assert store.deleted_chunk_ids == ["similar-doc:0", "similar-doc:1"]
