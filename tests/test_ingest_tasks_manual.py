from pathlib import Path
import shutil
import tempfile
import time

from rag_db.api.schemas import DocumentIngestRequest
from rag_db.config import Settings
from rag_db.document_loader import DocumentLoader
from rag_db.embedding_client import EmbeddingClient
from rag_db.rag import IngestionPipeline
from rag_db.services.document_ingestion import DocumentIngestionService
from rag_db.services.ingest_tasks import DocumentIngestTaskService
from rag_db.vector_store.base import VectorStore


class FakeVectorStore(VectorStore):
    def upsert(self, *, collection_name: str, chunks, embeddings) -> None:
        return None

    def search(self, *, collection_name: str, query_embedding: list[float], top_k: int = 5):
        return []

    def delete(self, *, collection_name: str, chunk_ids: list[str]) -> None:
        return None

    def find_duplicate_by_content_hash(self, *, collection_name: str, content_hash: str):
        return None

    def upsert_document_record(
        self,
        *,
        collection_name: str,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, object],
    ) -> None:
        return None

    def delete_document_record(self, *, collection_name: str, document_id: str) -> None:
        return None

    def find_similar_document(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ):
        return None


class FakeEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], int, str]:
        time.sleep(0.05)
        return [[0.1, 0.2] for _ in texts], 2, "fake-model"

    def embed_documents_with_progress(self, texts, *, progress_callback=None):
        embeddings = []
        total = len(texts)
        for index, _ in enumerate(texts, start=1):
            time.sleep(0.02)
            embeddings.append([0.1, 0.2])
            if progress_callback is not None:
                progress_callback(index, total)
        return embeddings, 2, "fake-model"


def test_task_service_completes_in_background() -> None:
    base = Path(tempfile.mkdtemp(dir="D:/scripty/rag_db/tests"))
    try:
        source = base / "sample.txt"
        source.write_text("one two three four five six seven eight nine ten", encoding="utf-8")

        settings = Settings(
            data_dir=base,
            chroma_dir=base / "chroma",
            chunk_size=12,
            chunk_overlap=3,
        )
        ingestion_service = DocumentIngestionService(
            settings=settings,
            loader=DocumentLoader(),
            embedding_client=FakeEmbeddingClient(),
            pipeline=IngestionPipeline(FakeVectorStore()),
        )
        task_service = DocumentIngestTaskService(ingestion_service)

        task = task_service.create_task(
            DocumentIngestRequest(
                source=str(source),
                collection_name="docs",
                document_id="doc-task-1",
            )
        )
        assert task.status in {"pending", "running"}

        deadline = time.time() + 3
        latest = task
        while time.time() < deadline:
            latest = task_service.get_task(task.task_id)
            if latest.status == "completed":
                break
            time.sleep(0.05)

        assert latest.status == "completed"
        assert latest.result is not None
        assert latest.result.document_id == "doc-task-1"
        assert latest.progress_percent == 100
    finally:
        shutil.rmtree(base, ignore_errors=True)
