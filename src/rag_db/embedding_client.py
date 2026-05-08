from __future__ import annotations

from pathlib import Path
from typing import Callable
import sys

from rag_db.config import Settings


class EmbeddingClient:
    """Thin wrapper around the sibling embedding_engine SDK."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        create_embedding_sdk = _load_create_embedding_sdk(self.settings)
        self._sdk = create_embedding_sdk(
            model=self.settings.embedding_model,
            device=self.settings.embedding_device,
            max_length=self.settings.embedding_max_length,
            cache_dir=self.settings.embedding_cache_dir,
            batch_size=self.settings.embedding_batch_size,
            preload=True,
        )

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], int, str]:
        return self.embed_documents_with_progress(texts)

    def embed_documents_with_progress(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[list[float]], int, str]:
        if not texts:
            return [], 0, self.settings.embedding_model

        all_embeddings: list[list[float]] = []
        dimension = 0
        model_name = self.settings.embedding_model
        total_batches = self._calculate_total_batches(len(texts))

        for batch_index, batch_texts in enumerate(self._iter_batches(texts), start=1):
            response = self._sdk.embed_texts(
                texts=batch_texts,
                type="document",
                output_dimension=self.settings.embedding_output_dimension,
            )
            all_embeddings.extend(response.embeddings)
            dimension = response.dimension
            model_name = response.model_name
            if progress_callback is not None:
                progress_callback(batch_index, total_batches)

        return all_embeddings, dimension, model_name

    def _iter_batches(self, texts: list[str]):
        batch_size = self._effective_batch_size(len(texts))
        for start in range(0, len(texts), batch_size):
            yield texts[start : start + batch_size]

    def _calculate_total_batches(self, text_count: int) -> int:
        batch_size = self._effective_batch_size(text_count)
        return (text_count + batch_size - 1) // batch_size

    def _effective_batch_size(self, text_count: int) -> int:
        if self.settings.embedding_batch_size is not None:
            return max(1, self.settings.embedding_batch_size)
        if text_count <= 8:
            return 1
        return 8


def _load_create_embedding_sdk(settings: Settings):
    last_error: Exception | None = None
    try:
        from embedding_engine import create_embedding_sdk

        return create_embedding_sdk
    except ModuleNotFoundError as exc:
        last_error = exc
        candidate_paths = [settings.embedding_engine_src, _default_embedding_engine_src()]
        for candidate in candidate_paths:
            if candidate is None or not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                from embedding_engine import create_embedding_sdk

                return create_embedding_sdk
            except ModuleNotFoundError as exc:
                last_error = exc
                continue
    message = (
        "embedding_engine is not importable. "
        "Set RAG_DB_EMBEDDING_ENGINE_SRC to its src directory and ensure its runtime "
        "dependencies are installed in the API environment."
    )
    if last_error is not None:
        message = f"{message} Root cause: {last_error}"
    raise ModuleNotFoundError(message)


def _default_embedding_engine_src() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "embedding_engine" / "src"
