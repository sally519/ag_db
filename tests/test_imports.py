from rag_db import Settings
from rag_db.chunking import TextChunker
from rag_db.models import DocumentChunk, SearchResult


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.env
    assert settings.log_level


def test_models_construct() -> None:
    chunk = DocumentChunk(chunk_id="c1", document_id="d1", content="hello")
    result = SearchResult(chunk_id="c1", score=0.9, content="hello")
    assert chunk.chunk_id == "c1"
    assert result.score == 0.9


def test_chunker_splits_text() -> None:
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.split_text("alpha beta gamma delta")
    assert chunks
    assert len(chunks) >= 2
