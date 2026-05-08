from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


def _read_optional_int(env_name: str) -> int | None:
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return None
    return int(raw)


def _read_optional_path(env_name: str) -> Path | None:
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return None
    return Path(raw)


@dataclass(slots=True)
class Settings:
    env: str = os.getenv("RAG_DB_ENV", "dev")
    data_dir: Path = Path(os.getenv("RAG_DB_DATA_DIR", "./data"))
    log_level: str = os.getenv("RAG_DB_LOG_LEVEL", "INFO")
    chroma_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RAG_DB_CHROMA_DIR", "./data/chroma"))
    )
    default_collection: str = os.getenv("RAG_DB_DEFAULT_COLLECTION", "documents")
    chunk_size: int = int(os.getenv("RAG_DB_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_DB_CHUNK_OVERLAP", "200"))
    embedding_model: str = os.getenv("RAG_DB_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_device: str = os.getenv("RAG_DB_EMBEDDING_DEVICE", "auto")
    embedding_max_length: int = int(os.getenv("RAG_DB_EMBEDDING_MAX_LENGTH", "2048"))
    embedding_batch_size: int | None = _read_optional_int("RAG_DB_EMBEDDING_BATCH_SIZE")
    embedding_cache_dir: str | None = os.getenv("RAG_DB_EMBEDDING_CACHE_DIR")
    embedding_output_dimension: int | None = _read_optional_int(
        "RAG_DB_EMBEDDING_OUTPUT_DIMENSION"
    )
    embedding_engine_src: Path | None = field(
        default_factory=lambda: _read_optional_path("RAG_DB_EMBEDDING_ENGINE_SRC")
    )

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.resolve()
        self.chroma_dir = self.chroma_dir.resolve()
        if self.embedding_engine_src is not None:
            self.embedding_engine_src = self.embedding_engine_src.resolve()
