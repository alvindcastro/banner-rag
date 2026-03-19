"""
app/azure_clients.py
Singleton Azure client factories — imported once, reused everywhere.
"""
import logging
from functools import lru_cache

from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    """Return a cached Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    """Return a cached Azure AI Search client (sync)."""
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


@lru_cache(maxsize=1)
def get_index_client() -> SearchIndexClient:
    """Return a cached Azure AI Search index management client."""
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


def embed_text(text: str) -> list[float]:
    """
    Embed a single string using Azure OpenAI text-embedding-ada-002.
    Returns a 1536-dim vector.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        input=[text],
        model=settings.azure_openai_embedding_deployment,
    )
    return response.data[0].embedding
