# Repository Guidelines

## Project Structure & Module Organization
This repository implements a local RAG ingestion service with `FastAPI`, `Chroma`, and a sibling `embedding_engine`.

- `src/rag_db/app.py`: FastAPI entrypoint and route registration.
- `src/rag_db/api/`: request/response schemas, dependencies, and HTTP routes.
- `src/rag_db/services/`: end-to-end workflows such as document ingestion.
- `src/rag_db/vector_store/`: vector store abstractions and Chroma implementation.
- `src/rag_db/rag/`: ingestion and retrieval pipelines.
- `src/rag_db/document_loader.py` and `chunking.py`: source loading and text splitting.
- `tests/`: pytest-based unit tests.
- `data/` and `logs/`: local runtime data; do not commit generated contents.

## Build, Test, and Development Commands
- `pip install -e .[dev]`: install the project in editable mode with dev tools.
- `pytest`: run the full test suite.
- `pytest tests/test_document_ingestion_service.py`: run a focused test file.
- `python -m compileall src tests`: quick syntax validation without starting services.
- `uvicorn rag_db.app:app --host 127.0.0.1 --port 8001 --app-dir src`: run the API locally.

If `embedding_engine` is not installed in the same environment, set `RAG_DB_EMBEDDING_ENGINE_SRC` to its `src` directory.

## Coding Style & Naming Conventions
- Use Python 3.11+ features and full type annotations.
- Use 4-space indentation and keep lines within Ruff’s 100-character limit.
- Prefer `snake_case` for functions, variables, and modules; use `PascalCase` for classes.
- Keep route handlers thin; put orchestration in `services/` and storage logic in `vector_store/`.
- Run `ruff check src/ tests/` and `ruff format src/ tests/` before submitting changes when available.

## Testing Guidelines
- Use `pytest` for all tests.
- Name test files `test_*.py` and test functions `test_*`.
- Add focused unit tests for new loaders, chunking rules, and vector-store-facing behavior.
- Prefer deterministic tests with fakes over tests that require live model downloads or network access.

## Commit & Pull Request Guidelines
Git history is not currently reliable to inspect from this environment because of repository ownership restrictions, so use simple imperative commit messages such as `Add document ingest endpoint` or `Fix file URL handling`.

For pull requests, include:
- a short summary of the behavior change,
- local verification steps you ran,
- any new environment variables or dependency changes,
- sample request/response payloads when API behavior changes.

## Security & Configuration Tips
- Keep secrets and tokens out of source control; use `.env` or environment variables.
- Review `.env.example` before running locally.
- Local and remote document ingestion may process untrusted files; validate supported file types before extending loaders.
