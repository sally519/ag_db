from __future__ import annotations

from pathlib import Path
from typing import Callable
import sys

from rag_db.config import Settings


class EmbeddingClient:
    """对 `embedding_engine` 的轻量封装。

    这个类的职责不是重新实现 embedding 能力，而是把兄弟项目暴露出的 SDK
    适配成当前 RAG 服务更容易使用的接口：
    - 查询向量化
    - 文档向量化
    - 带进度回调的批量向量化
    - 在导入失败时自动尝试本地源码路径兜底
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 embedding SDK，并在启动时预加载模型。"""
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
        """对外暴露的文档向量化入口。

        当前默认复用带进度回调的批量实现，便于后续统一维护批次策略。
        """
        return self.embed_documents_with_progress(texts)

    def embed_queries(
        self,
        texts: list[str],
        *,
        task_description: str | None = None,
    ) -> tuple[list[list[float]], int, str]:
        """将查询语句编码为向量。

        `embedding_engine` 会按 query 模式处理文本，从而与文档向量空间保持一致。
        """
        if not texts:
            return [], 0, self.settings.embedding_model

        response = self._sdk.embed_texts(
            texts=texts,
            type="query",
            output_dimension=self.settings.embedding_output_dimension,
            task_description=task_description,
        )
        return response.embeddings, response.dimension, response.model_name

    def embed_documents_with_progress(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[list[float]], int, str]:
        """按批量方式对文档列表进行向量化，并在批次完成后上报进度。

        `progress_callback` 的两个参数分别表示：
        - 当前已完成的批次数
        - 总批次数
        """
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
        """按当前有效批次大小切分输入文本列表。"""
        batch_size = self._effective_batch_size(len(texts))
        for start in range(0, len(texts), batch_size):
            yield texts[start : start + batch_size]

    def _calculate_total_batches(self, text_count: int) -> int:
        """根据文本总数和批次大小计算总批次数。"""
        batch_size = self._effective_batch_size(text_count)
        return (text_count + batch_size - 1) // batch_size

    def _effective_batch_size(self, text_count: int) -> int:
        """确定本次调用应使用的真实批次大小。

        如果显式配置了环境变量，则优先使用配置值；
        否则对小批量文本采用更保守的单条处理，避免初始化阶段内存波动过大。
        """
        if self.settings.embedding_batch_size is not None:
            return max(1, self.settings.embedding_batch_size)
        if text_count <= 8:
            return 1
        return 8


def _load_create_embedding_sdk(settings: Settings):
    """导入 `create_embedding_sdk`，并在必要时尝试本地源码路径兜底。"""
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
    """推断同级目录下 `embedding_engine/src` 的默认位置。"""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "embedding_engine" / "src"
