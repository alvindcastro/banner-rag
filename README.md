# Ellucian Banner Upgrade RAG Assistant

An Internal Knowledge Assistant (RAG System) that answers questions about Ellucian Banner upgrades using Azure OpenAI + Azure AI Search.

---

## Architecture

```
Your Docs (PDFs/TXT)
        │
        ▼
  scripts/ingest.py          ← Chunk, embed, and index your Banner docs
        │
        ▼
  Azure AI Search Index      ← Vector store of your knowledge
        │
        ▼
  app/main.py (FastAPI)      ← REST API
        │
        ▼
  /ask  endpoint             ← You POST a question, get a grounded answer
```

**Stack:**
- Python 3.11+
- Azure OpenAI (GPT-4o + text-embedding-ada-002)
- Azure AI Search (vector + keyword hybrid search)
- FastAPI + Uvicorn
- PyMuPDF (PDF parsing)
- LangChain (optional orchestration layer)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your Azure credentials in .env
```

### 3. Create the Azure AI Search index
```bash
python scripts/create_index.py
```

### 4. Ingest your Banner documents
```bash
# Drop PDFs or .txt files into data/docs/
python scripts/ingest.py
```

### 5. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

---

## API Endpoints

### `POST /ask`
Ask a question about Banner upgrades.

**Request:**
```json
{
  "question": "What changed in Banner Finance 9.3.22?",
  "top_k": 5,
  "version_filter": "9.3.22"
}
```

**Response:**
```json
{
  "answer": "In Banner Finance 9.3.22, the following changes were made...",
  "sources": [
    {
      "filename": "Banner_Finance_9.3.22_ReleaseNotes.pdf",
      "page": 4,
      "chunk": "...",
      "score": 0.92
    }
  ],
  "question": "What changed in Banner Finance 9.3.22?"
}
```

### `POST /ingest`
Trigger re-ingestion of documents in the data/docs/ folder.

### `GET /health`
Health check.

### `GET /index/stats`
Show document count and index status.

---

## Query Examples

```bash
# Ask about a specific Banner module upgrade
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the prerequisites for upgrading Banner Student to 9.39?"}'

# Filter by version
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Any known issues with this release?", "version_filter": "9.3.22"}'

# Ask about compatibility
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is Banner General compatible with Oracle 19c?"}'
```

---

## Adding Knowledge

Just drop files into `data/docs/` and re-run `python scripts/ingest.py`:

| Supported Formats | Notes |
|---|---|
| `.pdf` | Banner release notes, patch guides |
| `.txt` | Extracted text, runbooks |
| `.md` | Internal documentation |

Recommended naming: `Banner_<Module>_<Version>_ReleaseNotes.pdf`

---

## Azure Resources Required

| Resource | Tier | Purpose |
|---|---|---|
| Azure OpenAI | Standard | GPT-4o (chat) + ada-002 (embeddings) |
| Azure AI Search | Basic+ | Vector index |
| (Optional) Azure Blob Storage | LRS | Document archive |
