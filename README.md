# 🔍 RAG Pipeline

A modular, easy-to-configure **Retrieval-Augmented Generation** pipeline. Drop in your markdown documents, plug in your LLM provider, and get an expert Q&A / help system for your application — out of the box.

Built with **pgvector** (384-dimension embeddings), **sentence-transformers**, and **FastAPI**.

---

## Architecture

```
rag-pipeline/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── core/
│   │   ├── llm_provider.py  # Abstract LLM interface
│   │   └── groq_provider.py # Groq implementation
│   ├── database/
│   │   └── connection.py    # Postgres + pgvector connection & schema
│   └── services/
│       ├── ingest.py        # Document chunking & embedding pipeline
│       └── query.py         # Semantic search + LLM answer generation
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

### How It Works

```
Markdown Docs ──► Chunking ──► Embedding (384d) ──► pgvector Store
                                                        │
User Query ──► Embed Query ──► Cosine Similarity Search ┘
                                       │
                               Top-K Chunks ──► LLM ──► Answer + Sources
```

1. **Ingest** — Markdown files are split by headers, embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), and stored in Postgres via pgvector.
2. **Query** — User questions are embedded with the same model, matched against stored chunks using cosine similarity (`<=>` operator), and the top-K results are passed as context to the LLM.
3. **Generate** — The LLM synthesizes an answer grounded in the retrieved context and returns it alongside the source chunks.

---

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/StereoSachiiii/rag-pipeline.git
cd rag-pipeline
cp .env.example .env
```

Edit `.env` with your credentials:

```env
GROQ_API_KEY=your_groq_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

DB_HOST=localhost
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_PORT=5433
```

### 2. Run with Docker

```bash
docker compose up --build
```

This spins up:
- **Postgres + pgvector** on port `5433`
- **FastAPI app** on port `8000`

### 3. Initialize the Database

```bash
python -m app.database.connection
```

This creates the `pgvector` extension, the `documents` table with a `VECTOR(384)` column, and an **HNSW index** (`vector_cosine_ops`) for fast approximate nearest-neighbor search.

### 4. Ingest Documents

Place your `.md` files in the configured docs directory, then run:

```bash
python -m app.services.ingest
```

### 5. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "How do I reset my password?"}'
```

**Response:**

```json
{
  "answer": "To reset your password, navigate to Settings > Account...",
  "results": [
    {
      "content": "## Password Reset\nGo to Settings > Account...",
      "metadata": {"source": "user_guide.md", "headers": {"Section": "Password Reset"}}
    }
  ]
}
```

---

## Swapping the LLM Provider

The pipeline uses a provider pattern. To add your own LLM (OpenAI, Ollama, etc.), implement the `LLMProvider` interface:

```python
# app/core/llm_provider.py
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, context: str, question: str) -> str:
        pass
```

Then create your implementation:

```python
# app/core/my_provider.py
from app.core.llm_provider import LLMProvider

class MyProvider(LLMProvider):
    def generate(self, context: str, question: str) -> str:
        # call your LLM here
        return answer
```

And wire it up in `app/main.py`:

```python
rag_query = RAGQuery(llm_provider=MyProvider())
```

---

## Configuration

All configuration is done via environment variables (`.env`):

| Variable           | Description                        | Default                                    |
|--------------------|------------------------------------|--------------------------------------------|
| `GROQ_API_KEY`     | API key for Groq LLM              | —                                          |
| `EMBEDDING_MODEL`  | Sentence-transformer model name    | `sentence-transformers/all-MiniLM-L6-v2`   |
| `DB_HOST`          | Postgres host                      | `localhost`                                 |
| `DB_NAME`          | Postgres database name             | `ragdb`                                    |
| `DB_USER`          | Postgres user                      | `rag`                                      |
| `DB_PASSWORD`      | Postgres password                  | `rag`                                      |
| `DB_PORT`          | Postgres port                      | `5433`                                     |

> **Note:** The default embedding model produces **384-dimensional** vectors. If you swap the model, update the `VECTOR(384)` dimension in `app/database/connection.py` accordingly.

---

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| API Framework   | FastAPI                             |
| Vector Store    | PostgreSQL + pgvector               |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2, 384d) |
| LLM             | Groq (llama-3.1-8b-instant)         |
| Text Splitting  | LangChain `MarkdownHeaderTextSplitter` |
| Containerization| Docker & Docker Compose             |

---

## License

MIT
