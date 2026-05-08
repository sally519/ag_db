from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


def _read_optional_int(env_name: str) -> int | None:
    """读取可选整数环境变量。

    当变量不存在或为空字符串时返回 `None`，便于上层继续走默认逻辑。
    """
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return None
    return int(raw)


def _read_optional_path(env_name: str) -> Path | None:
    """读取可选路径环境变量。"""
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return None
    return Path(raw)


@dataclass(slots=True)
class Settings:
    """集中管理服务运行配置。

    配置主要来自环境变量，覆盖以下几类信息：
    - 服务环境与日志级别
    - Chroma 数据目录
    - 文本切块参数
    - embedding 模型、设备、缓存和输出维度
    - 本地 `embedding_engine` 源码路径
    """
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
    max_ingest_concurrency: int = int(os.getenv("RAG_DB_MAX_INGEST_CONCURRENCY", "5"))
    max_search_concurrency: int = int(os.getenv("RAG_DB_MAX_SEARCH_CONCURRENCY", "5"))
    embedding_engine_src: Path | None = field(
        default_factory=lambda: _read_optional_path("RAG_DB_EMBEDDING_ENGINE_SRC")
    )

    def __post_init__(self) -> None:
        """标准化路径配置，避免相对路径在不同启动目录下行为不一致。"""
        self.data_dir = self.data_dir.resolve()
        self.chroma_dir = self.chroma_dir.resolve()
        if self.embedding_engine_src is not None:
            self.embedding_engine_src = self.embedding_engine_src.resolve()
