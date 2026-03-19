"""
app/models.py
Pydantic schemas for API request and response payloads.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ─── Request Models ──────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Natural language question about Banner upgrades")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to retrieve")
    version_filter: Optional[str] = Field(
        default=None,
        description="Optional Banner version to filter results (e.g. '9.3.22')"
    )
    module_filter: Optional[str] = Field(
        default=None,
        description="Optional Banner module to filter results (e.g. 'Finance', 'Student')"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the prerequisites for Banner Finance 9.3.22?",
                "top_k": 5,
                "version_filter": "9.3.22",
                "module_filter": "Finance"
            }
        }
    }


class IngestRequest(BaseModel):
    docs_path: Optional[str] = Field(
        default="data/docs",
        description="Path to folder containing documents to ingest"
    )
    overwrite: bool = Field(
        default=False,
        description="If True, delete and re-create the index before ingesting"
    )


# ─── Response Models ─────────────────────────────────────────────

class SourceChunk(BaseModel):
    filename: str
    page: Optional[int] = None
    banner_module: Optional[str] = None
    banner_version: Optional[str] = None
    chunk_text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    retrieval_count: int


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    index_name: str
    chat_model: str
    embedding_model: str


class BlobSyncRequest(BaseModel):
    container_name: Optional[str] = Field(default=None, description="Override the default blob container name")
    prefix: Optional[str] = Field(default="", description="Only sync blobs with this prefix")
    overwrite: bool = Field(default=False, description="Re-download blobs that already exist locally")
    ingest_after_sync: bool = Field(default=True, description="Automatically run ingestion after blob sync")


class BlobSyncResponse(BaseModel):
    status: str
    files_downloaded: int
    downloaded_paths: list[str]
    ingestion: Optional[IngestResponse] = None
    message: str


class BlobListResponse(BaseModel):
    container_name: str
    document_count: int
    documents: list[dict]


class IndexStatsResponse(BaseModel):
    index_name: str
    document_count: int
    status: str
