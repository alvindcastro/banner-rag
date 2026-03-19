"""
scripts/create_index.py
Creates (or recreates) the Azure AI Search index with:
  - Full-text fields  (filename, module, version, chunk_text)
  - A vector field    (content_vector — 1536 dims for ada-002)
  - Filterable fields (banner_module, banner_version)

Run once before ingesting documents:
    python scripts/create_index.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from rich.console import Console

from app.azure_clients import get_index_client
from app.config import settings

console = Console()

EMBEDDING_DIMENSIONS = 1536  # text-embedding-ada-002


def create_banner_index(overwrite: bool = False) -> None:
    index_client = get_index_client()
    index_name = settings.azure_search_index_name

    # Delete existing index if overwrite requested
    if overwrite:
        try:
            index_client.delete_index(index_name)
            console.print(f"[yellow]Deleted existing index: {index_name}[/yellow]")
        except Exception:
            pass  # Index didn't exist

    fields = [
        # Unique document chunk ID
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # Source document filename
        SimpleField(
            name="filename",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        # Page number within PDF
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        # Banner module (Finance, Student, HR, General, etc.)
        SimpleField(
            name="banner_module",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # Banner version (e.g. 9.3.22)
        SimpleField(
            name="banner_version",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # The actual text content of this chunk (BM25 searchable)
        SearchableField(
            name="chunk_text",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        # 1536-dim embedding vector (ada-002)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="banner-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="banner-hnsw",
                parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="banner-vector-profile",
                algorithm_configuration_name="banner-hnsw",
            )
        ],
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )

    index_client.create_or_update_index(index)
    console.print(f"[green]✓ Index '{index_name}' created/updated successfully.[/green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create Banner RAG search index")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    console.print("[bold]Creating Azure AI Search index for Banner RAG...[/bold]")
    create_banner_index(overwrite=args.overwrite)
