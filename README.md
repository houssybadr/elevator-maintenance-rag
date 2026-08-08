# Elevator Maintenance RAG

This project is a **Retrieval-Augmented Generation (RAG)** system for elevator maintenance teams.
It helps technicians ask operational questions and receive answers grounded in technical manuals.

## Project Structure

- [`rag/app/`](C:/Users/user/Desktop/rag/rag/app): FastAPI RAG API (query + health endpoints).
- [`rag/data/`](C:/Users/user/Desktop/rag/rag/data): document ingestion/chunking/embedding/indexing scripts.
- [`vector_database/`](C:/Users/user/Desktop/rag/vector_database): PostgreSQL + pgvector build and DB schema.
- [`docker-compose.yaml`](C:/Users/user/Desktop/rag/docker-compose.yaml): local orchestration (API + vector DB).

## Main Features

### 1) RAG Question Answering API
- `POST /query`
  - Accepts a technician question.
  - Creates embedding from the question.
  - Retrieves top relevant chunks from PostgreSQL/pgvector.
  - Builds a prompt and generates an answer with the LLM.
  - Returns:
    - `answer`
    - `token_usage`

### 2) Secure API Access
- API key protection via `X-API-Key` header.
- Built-in rate limiting (SlowAPI).

### 3) Advanced Health Check
- `GET /health` verifies:
  - internet connectivity
  - PostgreSQL connectivity
  - pgvector extension availability
  - `documents` table existence
  - embedder model loaded state

### 4) Generic Multi-Document Ingestion Pipeline
The script [`ingest_documents_pipeline.py`](C:/Users/user/Desktop/rag/rag/data/ingest_documents_pipeline.py) can ingest **new manuals dynamically** (different brands/models/types):

- Supported input formats: `.pdf`, `.txt`, `.md`, `.markdown`
- Automatic flow:
  1. Read & parse documents
  2. Split into RAG chunks
  3. Generate embeddings (BGE-M3)
  4. Insert into pgvector database

Metadata stored per chunk (for better filtering):
- `brand`
- `elevator_model`
- `document_type`
- `document_version`
- `added_at`
- `page_number`
- `section`
- `source_file`
- `source_path`
- `format`
- `language`
- `embedding_model`
- `chunk_index`
- `chunk_total`

## Ingestion Usage

### A) Ingest files/folders with shared metadata
```bash
python rag/data/ingest_documents_pipeline.py \
  --inputs rag/data/documents/manual.pdf rag/data/documents/troubleshooting.txt \
  --brand Hyundai \
  --elevator-model NEXIEZ \
  --document-type maintenance_manual \
  --document-version v3 \
  --output-json rag/data/documents/chunks_with_embeddings.json
```

### B) Ingest with per-document metadata manifest
```bash
python rag/data/ingest_documents_pipeline.py --manifest rag/data/documents/ingestion_manifest.json
```

Example manifest:
```json
[
  {
    "path": "rag/data/documents/manual.pdf",
    "metadata": {
      "brand": "Hyundai",
      "elevator_model": "NEXIEZ",
      "document_type": "maintenance_manual",
      "document_version": "v3"
    }
  },
  {
    "path": "rag/data/documents/guide.txt",
    "metadata": {
      "brand": "Otis",
      "elevator_model": "Gen2",
      "document_type": "troubleshooting_guide",
      "document_version": "2026-01"
    }
  }
]
```

## Local Run (Docker)

```bash
docker compose up --build
```

Services:
- PostgreSQL/pgvector
- RAG API on port `8000`

## Configuration

Set required environment values in:
- [`rag/app/.env`](C:/Users/user/Desktop/rag/rag/app/.env)
- [`rag/data/vecdb.secrets.env`](C:/Users/user/Desktop/rag/rag/data/vecdb.secrets.env)
- [`vector_database/vecdb.secrets.env`](C:/Users/user/Desktop/rag/vector_database/vecdb.secrets.env)

Key variables include:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `API_KEY`
- `GOOGLE_API_KEY`, `GROQ_API_KEY`
- `EMBEDDING_MODEL` (optional override)

## Continuous Integration

A GitHub Actions workflow is configured at:
- [`.github/workflows/ci.yml`](C:/Users/user/Desktop/rag/.github/workflows/ci.yml)

On every **push** and **pull request**, it:
- installs dependencies
- starts a PostgreSQL service with pgvector
- initializes the `documents` schema
- starts the FastAPI app in CI-safe mode (`SKIP_EMBEDDER_LOAD=true`)
- calls `GET /health` and fails if any required check is not `ok`

## Git Ignore Policy for Document Corpora

To avoid committing heavy or private manuals, all `documents` folders are ignored by Git (including [`rag/data/documents/`](C:/Users/user/Desktop/rag/rag/data/documents)).
