"""
app/config.py
Central configuration loaded from environment / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str = "banner-upgrade-knowledge"

    # RAG tuning
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_default: int = 5

    # Azure Blob Storage (optional — for pulling PDFs from blob)
    azure_storage_connection_string: str = ""
    azure_storage_container_name: str = "banner-release-notes"
    azure_storage_blob_prefix: str = ""     # Optional subfolder prefix

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


# Singleton
settings = Settings()
