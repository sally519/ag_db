# rag_db

一个面向向量数据库与 RAG 查询实验的 Python 工程骨架。

## 当前初始化内容

- 使用 `src/` 布局组织 Python 包
- 预留配置、数据模型、向量存储抽象、文档入库、检索流水线模块
- 附带基础测试与开发依赖定义

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```

## 目录结构

```text
rag_db/
├─ src/rag_db/
│  ├─ vector_store/
│  └─ rag/
├─ tests/
└─ pyproject.toml
```

## 后续建议方向

1. 明确底层向量存储实现：本地内存、FAISS、Chroma、Milvus、pgvector。
2. 定义文档切分、embedding、索引构建和召回接口。
3. 再接入具体 LLM/Reranker 形成完整 RAG 链路。

