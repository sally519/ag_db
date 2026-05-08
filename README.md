# rag_db

基于 `FastAPI + Chroma + embedding_engine` 的本地 RAG 向量化存储服务。

## 当前能力

- 接收本地文件路径、`file://` 路径或 HTTP/HTTPS URL
- 抽取文本后进行分块、向量化，并写入本地 Chroma
- 当前优先支持 `txt / md / json / html / pdf`
- 使用兄弟工程 `D:\scripty\embedding_engine` 作为 embedding 后端
- 文档入库采用异步任务模式，前端可轮询查询进度

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

如果 `embedding_engine` 没有安装到当前环境，配置：

```powershell
$env:RAG_DB_EMBEDDING_ENGINE_SRC="D:\scripty\embedding_engine\src"
```

## 启动服务

```powershell
uvicorn rag_db.app:app --host 127.0.0.1 --port 8001 --app-dir src
```

## 接口说明

所有业务接口均使用 `POST` 并通过 JSON 传输。

### 1. 创建入库任务

`POST /api/documents/ingest`

请求示例：

```json
{
  "source": "D:/scripty/rag_db/关于开展数字化总平台业务线2025年度评优评先工作的通知.pdf",
  "collection_name": "company_docs",
  "document_id": "notice-2025-01",
  "duplicate_strategy": "skip",
  "similarity_threshold": 0.98,
  "similarity_strategy": "skip",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "metadata": {
    "tenant": "default",
    "biz": "notice"
  }
}
```

响应示例：

```json
{
  "success": true,
  "task_id": "f0d29d9b4e5f4dc1a58fdca4947d3bf7",
  "status": "pending",
  "progress_percent": 0,
  "stage": "queued",
  "message": "Task queued"
}
```

### 2. 查询任务进度

`POST /api/documents/ingest/status`

请求示例：

```json
{
  "task_id": "f0d29d9b4e5f4dc1a58fdca4947d3bf7"
}
```

响应示例：

```json
{
  "success": true,
  "task_id": "f0d29d9b4e5f4dc1a58fdca4947d3bf7",
  "status": "running",
  "progress_percent": 70,
  "stage": "embedding",
  "message": "Generating embeddings",
  "error": null,
  "result": null
}
```

任务完成后，`result` 会带最终入库结果。

## 去重策略

`duplicate_strategy` 缺省值是 `skip`：

- `skip`：命中完全重复文档时跳过入库，返回已有文档信息
- `reject`：命中完全重复文档时任务失败，结果中会有错误信息
- `replace`：删除旧文档对应 chunks，再写入新文档

## 高相似策略

文档在精确去重之后，还会做文档级相似度检查。

- `similarity_threshold`：相似度阈值，默认 `0.98`
- `similarity_strategy`：`off / skip / reject / replace`

默认是：

```json
{
  "similarity_threshold": 0.98,
  "similarity_strategy": "skip"
}
```

含义是：如果新文档与库内已有文档的文档级相似度 `>= 0.98`，则默认跳过入库，避免高相似文档造成冗余。

## 查询接口

`POST /api/documents/search`

查询流程：

1. 先做向量召回，默认取 `10` 条
2. 再对这 `10` 条做 rerank
3. 最终默认返回 `3` 条

请求示例：

```json
{
  "query": "评优评先申报时间是什么",
  "recall_top_k": 10,
  "rerank_top_n": 3
}
```

响应示例：

```json
{
  "success": true,
  "query": "评优评先申报时间是什么",
  "searched_collections": [
    "company_docs",
    "policy_docs"
  ],
  "recall_top_k": 10,
  "rerank_top_n": 3,
  "hits": [
    {
      "chunk_id": "notice-2025-01:0",
      "collection_name": "company_docs",
      "content": "......",
      "metadata": {
        "document_id": "notice-2025-01"
      },
      "recall_score": 0.84,
      "rerank_score": 0.92
    }
  ]
}
```
