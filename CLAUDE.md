# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

rag_db 是一个面向向量数据库与 RAG（Retrieval-Augmented Generation）查询实验的 Python 工程骨架。使用 `src/` 布局，Python >= 3.11。

## 常用命令

```bash
# 安装（开发模式，含 dev 依赖）
pip install -e .[dev]

# 运行全部测试
pytest

# 运行单个测试文件/函数
pytest tests/test_imports.py
pytest tests/test_imports.py::test_settings_defaults

# Lint
ruff check src/ tests/

# 格式化
ruff format src/ tests/
```

## 架构

核心抽象分三层，依赖方向为上层依赖下层：

1. **models** (`src/rag_db/models.py`) — `DocumentChunk` 和 `SearchResult` 两个 dataclass，作为全项目通用的数据交换类型。
2. **vector_store** (`src/rag_db/vector_store/base.py`) — `VectorStore` ABC，定义 `upsert` / `search` / `delete` 三个抽象方法。后续具体实现（FAISS、Chroma、pgvector 等）继承此类。
3. **rag** (`src/rag_db/rag/`) — `IngestionPipeline` 和 `RetrievalPipeline`，组合使用 `VectorStore` 完成文档入库与检索。

配置集中在 `Settings` dataclass（`src/rag_db/config.py`），通过环境变量 `RAG_DB_ENV`、`RAG_DB_DATA_DIR`、`RAG_DB_LOG_LEVEL` 控制，默认值参见 `.env.example`。

## 代码风格

- Ruff 作为 linter/formatter，行宽 100，目标版本 py311
- 使用 `slots=True` 的 dataclass
- 类型注解全覆盖（Python 3.11+ 语法，如 `dict[str, Any]`、`list[...]`）
