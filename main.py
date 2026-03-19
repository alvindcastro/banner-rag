"""
app/main.py
FastAPI application exposing the Banner Upgrade RAG API.

Endpoints:
  GET  /health           — liveness check
  GET  /index/stats      — document count in search index
  POST /ask              — ask a question (full RAG pipeline)
  POST /ingest           — ingest docs from local data/docs/ folder
  POST /blob/sync        — download PDFs from Azure Blob + optionally ingest
  GET  /blob/list        — list documents available in blob storage
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    AskRequest, AskResponse,
    IngestRequest, IngestResponse,
    BlobSyncRequest, BlobSyncResponse, BlobListResponse,
    HealthResponse, IndexStatsResponse,
)
from app.rag import ask as rag_ask
from app.azure_clients import get_search_client, get_openai_client

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Banner RAG API starting up...")
    logger.info(f"   Chat model      : {settings.azure_openai_chat_deployment}")
    logger.info(f"   Embedding model : {settings.azure_openai_embedding_deployment}")
    logger.info(f"   Search index    : {settings.azure_search_index_name}")
    logger.info(f"   Blob container  : {settings.azure_storage_container_name}")
    yield
    logger.info("🛑 Banner RAG API shutting down.")


# ─── App Instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ellucian Banner Upgrade Knowledge Assistant",
    description=(
        "Internal RAG API for answering questions about Ellucian Banner ERP upgrades. "
        "Powered by Azure OpenAI (GPT-4o), Azure AI Search (vector + hybrid), "
        "and Azure Blob Storage (PDF source)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── System Routes ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Verify the API and all Azure connections are reachable."""
    try:
        get_openai_client()
        get_search_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Azure connection error: {exc}")
    return HealthResponse(
        status="healthy",
        index_name=settings.azure_search_index_name,
        chat_model=settings.azure_openai_chat_deployment,
        embedding_model=settings.azure_openai_embedding_deployment,
    )


@app.get("/index/stats", response_model=IndexStatsResponse, tags=["System"])
def index_stats():
    """Return document count and status for the Azure AI Search index."""
    try:
        client = get_search_client()
        count = client.get_document_count()
        return IndexStatsResponse(
            index_name=settings.azure_search_index_name,
            document_count=count,
            status="ready",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── RAG Route ────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(request: AskRequest):
    """
    Ask a natural language question about Banner upgrades.

    The system will:
    1. Embed the question using Azure OpenAI (ada-002)
    2. Run hybrid search (vector + BM25 keyword) against Azure AI Search
    3. Build a grounded prompt from the top-K retrieved chunks
    4. Generate an answer using GPT-4o
    5. Return the answer with source citations

    Use `version_filter` and `module_filter` to scope results to a
    specific Banner release or module.
    """
    try:
        logger.info(
            f"Q: {request.question!r} | "
            f"top_k={request.top_k} | "
            f"version={request.version_filter} | "
            f"module={request.module_filter}"
        )
        response = rag_ask(request)
        logger.info(f"A: {len(response.answer)} chars | {response.retrieval_count} sources retrieved")
        return response
    except Exception as exc:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Ingestion Routes ─────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents(request: IngestRequest):
    """
    Parse, chunk, embed, and index Banner documents from a local folder.

    Drop PDFs into `data/docs/` (or specify `docs_path`) and call this endpoint.
    Set `overwrite=true` to delete and recreate the search index from scratch.
    """
    try:
        from scripts.ingest import run_ingestion
        result = run_ingestion(
            docs_path=request.docs_path,
            overwrite=request.overwrite,
        )
        return IngestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/blob/list", response_model=BlobListResponse, tags=["Blob Storage"])
def list_blob_documents(prefix: str = ""):
    """
    List all Banner PDF documents available in Azure Blob Storage.
    Does NOT download or ingest — just shows what's there.
    """
    if not settings.azure_storage_connection_string:
        raise HTTPException(
            status_code=400,
            detail="AZURE_STORAGE_CONNECTION_STRING is not configured in .env"
        )
    try:
        from app.blob_storage import list_blob_documents as _list
        docs = _list(
            connection_string=settings.azure_storage_connection_string,
            container_name=settings.azure_storage_container_name,
            prefix=prefix or settings.azure_storage_blob_prefix,
        )
        return BlobListResponse(
            container_name=settings.azure_storage_container_name,
            document_count=len(docs),
            documents=docs,
        )
    except Exception as exc:
        logger.exception("Blob list error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/blob/sync", response_model=BlobSyncResponse, tags=["Blob Storage"])
def sync_blob_and_ingest(request: BlobSyncRequest):
    """
    Download Banner PDF release notes from Azure Blob Storage,
    then optionally run the full ingestion pipeline.

    **Typical workflow:**
    1. Upload Banner PDFs to your Azure Blob container.
    2. Call `POST /blob/sync` — downloads + ingests automatically.
    3. Call `POST /ask` to query the knowledge base.

    Set `ingest_after_sync=false` if you want to download only
    and run `/ingest` separately.
    """
    if not settings.azure_storage_connection_string:
        raise HTTPException(
            status_code=400,
            detail="AZURE_STORAGE_CONNECTION_STRING is not configured in .env"
        )

    try:
        from app.blob_storage import download_docs_from_blob
        from scripts.ingest import run_ingestion

        container = request.container_name or settings.azure_storage_container_name
        prefix = request.prefix if request.prefix is not None else settings.azure_storage_blob_prefix

        logger.info(f"Syncing blobs from container='{container}' prefix='{prefix}'")
        downloaded = download_docs_from_blob(
            connection_string=settings.azure_storage_connection_string,
            container_name=container,
            local_dest="data/docs",
            prefix=prefix,
            overwrite=request.overwrite,
        )

        ingestion_result = None
        if request.ingest_after_sync and downloaded:
            logger.info(f"Running ingestion on {len(downloaded)} downloaded files...")
            raw = run_ingestion(docs_path="data/docs", overwrite=request.overwrite)
            ingestion_result = IngestResponse(**raw)

        return BlobSyncResponse(
            status="success",
            files_downloaded=len(downloaded),
            downloaded_paths=downloaded,
            ingestion=ingestion_result,
            message=(
                f"Synced {len(downloaded)} files from '{container}'. "
                + (f"Indexed {ingestion_result.chunks_indexed} chunks." if ingestion_result else "Ingestion skipped.")
            ),
        )

    except Exception as exc:
        logger.exception("Blob sync error")
        raise HTTPException(status_code=500, detail=str(exc))
