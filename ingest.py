"""
scripts/ingest.py
Ingestion pipeline for Banner upgrade documentation.

Steps:
  1. Walk docs_path for .pdf, .txt, .md files
  2. Parse text from each file
  3. Split into overlapping chunks
  4. Embed each chunk via Azure OpenAI
  5. Upload chunk documents to Azure AI Search

Naming convention (optional — used to auto-detect module/version):
  Banner_<Module>_<Version>_ReleaseNotes.pdf
  e.g. Banner_Finance_9.3.22_ReleaseNotes.pdf

Run:
    python scripts/ingest.py                          # default: data/docs/
    python scripts/ingest.py --path /path/to/docs
    python scripts/ingest.py --overwrite              # recreates the index first
"""
import sys
import os
import re
import hashlib
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from rich.console import Console
from rich.progress import track

from app.azure_clients import get_search_client, embed_text
from app.config import settings
from scripts.create_index import create_banner_index

console = Console()
logger = logging.getLogger(__name__)

# ─── Text Extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page, text}."""
    pages = []
    doc = fitz.open(str(path))
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def extract_text_from_txt(path: Path) -> list[dict]:
    """Read a .txt or .md file as a single 'page'."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [{"page": 1, "text": text}] if text else []


def extract_pages(path: Path) -> list[dict]:
    """Dispatch extraction based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in (".txt", ".md"):
        return extract_text_from_txt(path)
    else:
        logger.warning(f"Unsupported file type: {path}")
        return []


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    Attempts to split on paragraph/sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a clean break point (paragraph > newline > space)
        for sep in ["\n\n", "\n", ". ", " "]:
            pos = text.rfind(sep, start, end)
            if pos > start:
                end = pos + len(sep)
                break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap

    return [c for c in chunks if c]


# ─── Metadata Parsing ────────────────────────────────────────────────────────

KNOWN_MODULES = [
    "Finance", "Student", "HR", "Human Resources",
    "Financial Aid", "General", "Advancement", "Payroll",
    "Accounts Receivable", "Position Control",
]

VERSION_PATTERN = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)")


def parse_metadata_from_filename(filename: str) -> dict:
    """
    Attempt to extract Banner module and version from filename.
    E.g. Banner_Finance_9.3.22_ReleaseNotes.pdf → module=Finance, version=9.3.22
    """
    module = None
    version = None

    for mod in KNOWN_MODULES:
        if mod.replace(" ", "_").lower() in filename.lower() or mod.lower() in filename.lower():
            module = mod
            break

    version_matches = VERSION_PATTERN.findall(filename)
    if version_matches:
        # Pick the first version-like number that isn't a year
        for v in version_matches:
            if not v.startswith("20"):  # Skip years like 2024
                version = v
                break

    return {"banner_module": module, "banner_version": version}


# ─── Ingestion ────────────────────────────────────────────────────────────────

def chunk_id(filename: str, page: int, chunk_index: int) -> str:
    """Generate a deterministic, URL-safe chunk ID."""
    raw = f"{filename}::p{page}::c{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_file(path: Path, search_client, chunk_size: int, chunk_overlap: int) -> int:
    """Parse, chunk, embed, and upload a single file. Returns chunk count."""
    meta = parse_metadata_from_filename(path.name)
    pages = extract_pages(path)
    if not pages:
        console.print(f"  [yellow]⚠ No text extracted from {path.name}[/yellow]")
        return 0

    documents = []
    chunk_index = 0

    for page_data in pages:
        chunks = chunk_text(page_data["text"], chunk_size, chunk_overlap)
        for chunk in chunks:
            vector = embed_text(chunk)
            doc_id = chunk_id(path.name, page_data["page"], chunk_index)
            documents.append({
                "id": doc_id,
                "filename": path.name,
                "page_number": page_data["page"],
                "banner_module": meta["banner_module"],
                "banner_version": meta["banner_version"],
                "chunk_text": chunk,
                "content_vector": vector,
            })
            chunk_index += 1

    # Upload in batches of 100 (Azure Search limit per batch)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        search_client.upload_documents(documents=batch)

    return len(documents)


def run_ingestion(docs_path: str = "data/docs", overwrite: bool = False) -> dict:
    """
    Main ingestion entry point.
    Called by both the CLI and the /ingest API endpoint.
    """
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs path does not exist: {docs_dir.resolve()}")

    if overwrite:
        console.print("[yellow]Recreating index...[/yellow]")
        create_banner_index(overwrite=True)

    supported = [".pdf", ".txt", ".md"]
    files = [f for f in docs_dir.rglob("*") if f.suffix.lower() in supported]

    if not files:
        return {
            "status": "warning",
            "documents_processed": 0,
            "chunks_indexed": 0,
            "message": f"No supported files found in {docs_dir}",
        }

    search_client = get_search_client()
    total_chunks = 0
    docs_processed = 0

    for file_path in track(files, description="Ingesting documents..."):
        console.print(f"  📄 {file_path.name}")
        try:
            n = ingest_file(
                path=file_path,
                search_client=search_client,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            total_chunks += n
            docs_processed += 1
            console.print(f"     [green]✓ {n} chunks indexed[/green]")
        except Exception as exc:
            console.print(f"     [red]✗ Error: {exc}[/red]")
            logger.exception(f"Failed to ingest {file_path}")

    return {
        "status": "success",
        "documents_processed": docs_processed,
        "chunks_indexed": total_chunks,
        "message": f"Ingested {docs_processed} documents ({total_chunks} chunks) into '{settings.azure_search_index_name}'",
    }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Banner documents into Azure AI Search")
    parser.add_argument("--path", default="data/docs", help="Folder containing Banner docs")
    parser.add_argument("--overwrite", action="store_true", help="Recreate the index before ingesting")
    args = parser.parse_args()

    console.rule("[bold blue]Banner RAG — Document Ingestion[/bold blue]")
    result = run_ingestion(docs_path=args.path, overwrite=args.overwrite)

    console.print()
    console.rule("Results")
    console.print(f"  Status            : [bold]{result['status']}[/bold]")
    console.print(f"  Documents         : {result['documents_processed']}")
    console.print(f"  Chunks indexed    : {result['chunks_indexed']}")
    console.print(f"  Message           : {result['message']}")
