from rag_db.models import SearchResult
from rag_db.services.reranker import EmbeddingReranker
from rag_db.services.search import DocumentSearchService


class FakeEmbeddingClient:
    def embed_queries(self, texts, *, task_description=None):
        return [[1.0, 0.0]], 2, "fake-model"

    def embed_documents(self, texts):
        mapping = {
            "alpha": [1.0, 0.0],
            "beta": [0.8, 0.2],
            "gamma": [0.1, 0.9],
        }
        return [mapping[text] for text in texts], 2, "fake-model"


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
