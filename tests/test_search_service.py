from threading import BoundedSemaphore

from rag_db.config import Settings
from rag_db.models import SearchResult
from rag_db.services.reranker import EmbeddingReranker
from rag_db.services.search import DocumentSearchService, SearchConcurrencyLimitError


class FakeEmbeddingClient:
    def embed_queries(self, texts, *, task_description=None):
        return [[1.0, 0.0]], 2, "fake-model"

    def rerank_documents(self, *, query, documents, top_n):
        score_map = {
            "alpha": 0.95,
            "beta": 0.7,
            "gamma": 0.1,
        }
        ranked = sorted(
            [
                {
                    "index": index,
                    "document": document,
                    "relevance_score": score_map[document],
                }
                for index, document in enumerate(documents)
            ],
            key=lambda item: item["relevance_score"],
            reverse=True,
        )
        return ranked[:top_n], "fake-reranker"


class FakeRetrievalPipeline:
    def list_searchable_collections(self) -> list[str]:
        return ["docs", "policies"]

    def retrieve(self, *, collection_name: str, query_embedding: list[float], top_k: int = 5):
        by_collection = {
            "docs": [
                SearchResult(chunk_id="c3", score=0.7, content="gamma", metadata={"rank": 3}),
                SearchResult(chunk_id="c2", score=0.3, content="beta", metadata={"rank": 2}),
                SearchResult(chunk_id="c1", score=0.2, content="alpha", metadata={"rank": 1}),
            ],
            "policies": [
                SearchResult(chunk_id="p1", score=0.4, content="beta", metadata={"rank": 4}),
                SearchResult(chunk_id="p2", score=0.1, content="alpha", metadata={"rank": 5}),
            ],
        }
        return by_collection[collection_name][:top_k]


def test_search_service_reranks_and_limits_results() -> None:
    service = DocumentSearchService(
        settings=Settings(),
        retrieval_pipeline=FakeRetrievalPipeline(),
        embedding_client=FakeEmbeddingClient(),
        reranker=EmbeddingReranker(FakeEmbeddingClient()),
    )

    hits = service.search(
        query="find alpha",
        recall_top_k=10,
        rerank_top_n=3,
    )

    assert [item.chunk_id for item in hits] == ["p2", "c1", "c2"]
    assert hits[0].rerank_score >= hits[1].rerank_score >= hits[2].rerank_score
    assert all(item.recall_score is not None for item in hits)
    assert {item.collection_name for item in hits} == {"docs", "policies"}


def test_search_service_rejects_when_concurrency_limit_is_reached() -> None:
    service = DocumentSearchService(
        settings=Settings(max_search_concurrency=1),
        retrieval_pipeline=FakeRetrievalPipeline(),
        embedding_client=FakeEmbeddingClient(),
        reranker=EmbeddingReranker(FakeEmbeddingClient()),
    )
    service._slots = BoundedSemaphore(value=1)
    assert service._slots.acquire(blocking=False) is True

    try:
        try:
            service.search(query="find alpha")
            assert False, "expected SearchConcurrencyLimitError"
        except SearchConcurrencyLimitError:
            pass
    finally:
        service._slots.release()
