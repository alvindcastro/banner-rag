"""
app/blob_storage.py
Azure Blob Storage integration.

Supports two workflows:
  1. download_docs_from_blob()  — sync all PDFs from a blob container to local data/docs/
  2. stream_blob_pdf()          — stream a single blob directly for in-memory parsing

Configure in .env:
    AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_SAS_TOKEN
    AZURE_STORAGE_CONTAINER_NAME
"""
import logging
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContainerClient
from rich.console import Console
from rich.progress import track

logger = logging.getLogger(__name__)
console = Console()


def _get_container_client(connection_string: str, container_name: str) -> ContainerClient:
    service_client = BlobServiceClient.from_connection_string(connection_string)
    return service_client.get_container_client(container_name)


def download_docs_from_blob(
    connection_string: str,
    container_name: str,
    local_dest: str = "data/docs",
    prefix: str = "",
    overwrite: bool = False,
) -> list[str]:
    """
    Download all PDF (and .txt/.md) files from an Azure Blob container
    to a local directory. Skips files that already exist unless overwrite=True.

    Args:
        connection_string: Azure Storage connection string
        container_name:    Blob container name (e.g. 'banner-release-notes')
        local_dest:        Local folder to download into
        prefix:            Optional blob prefix filter (e.g. 'finance/' or '2024/')
        overwrite:         Re-download even if local file exists

    Returns:
        List of local file paths downloaded
    """
    dest_path = Path(local_dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    container_client = _get_container_client(connection_string, container_name)

    supported_extensions = {".pdf", ".txt", ".md"}
    blobs = [
        b for b in container_client.list_blobs(name_starts_with=prefix)
        if Path(b.name).suffix.lower() in supported_extensions
    ]

    if not blobs:
        console.print(f"[yellow]No supported files found in blob container '{container_name}' (prefix='{prefix}')[/yellow]")
        return []

    console.print(f"[bold]Found {len(blobs)} documents in blob storage.[/bold]")
    downloaded = []

    for blob in track(blobs, description="Downloading from Azure Blob..."):
        # Flatten blob path to filename (strip directory prefix)
        local_filename = Path(blob.name).name
        local_path = dest_path / local_filename

        if local_path.exists() and not overwrite:
            console.print(f"  [dim]Skipping (already exists): {local_filename}[/dim]")
            continue

        blob_client = container_client.get_blob_client(blob.name)
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)

        console.print(f"  [green]✓ Downloaded:[/green] {local_filename}")
        downloaded.append(str(local_path))

    console.print(f"\n[green]Downloaded {len(downloaded)} new files to {dest_path.resolve()}[/green]")
    return downloaded


def list_blob_documents(
    connection_string: str,
    container_name: str,
    prefix: str = "",
) -> list[dict]:
    """
    List all documents in a blob container without downloading.
    Returns metadata about each blob.
    """
    container_client = _get_container_client(connection_string, container_name)
    supported_extensions = {".pdf", ".txt", ".md"}

    results = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if Path(blob.name).suffix.lower() in supported_extensions:
            results.append({
                "name": blob.name,
                "size_bytes": blob.size,
                "last_modified": str(blob.last_modified),
                "content_type": blob.content_settings.content_type if blob.content_settings else None,
            })
    return results
