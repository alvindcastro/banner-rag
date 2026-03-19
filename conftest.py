"""
conftest.py
Pytest configuration: mock Azure/OpenAI packages and wire up the flat-module
structure so tests can import rag, ingest, models, and main without real
Azure credentials.
"""
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. Required environment variables (must be set before config.py loads) ──
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
os.environ.setdefault("AZURE_SEARCH_API_KEY", "test-search-key")

# ── 2. Mock external packages unavailable in the test environment ────────────
_MOCK_PKGS = [
    "azure",
    "azure.search",
    "azure.search.documents",
    "azure.search.documents.models",
    "azure.search.documents.aio",
    "azure.search.documents.indexes",
    "azure.core",
    "azure.core.credentials",
    "azure.storage",
    "azure.storage.blob",
    "azure.identity",
    "openai",
    "fitz",
]
for _pkg in _MOCK_PKGS:
    sys.modules.setdefault(_pkg, MagicMock())

# ── 3. Create fake top-level packages for app.* / scripts.* ─────────────────
sys.modules.setdefault("app", types.ModuleType("app"))
sys.modules.setdefault("scripts", types.ModuleType("scripts"))


def _load(sys_name: str, file_path: str):
    """Load a source file as a named module and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(sys_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[sys_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 4. Load real config and models (no Azure deps) ───────────────────────────
_load("app.config", os.path.join(BASE, "config.py"))
_load("app.models", os.path.join(BASE, "models.py"))

# ── 5. Mock azure_clients (prevents real network calls) ──────────────────────
_mock_ac = MagicMock()
_mock_ac.embed_text.return_value = [0.0] * 1536
sys.modules["app.azure_clients"] = _mock_ac

# ── 6. Mock scripts.create_index ─────────────────────────────────────────────
sys.modules["scripts.create_index"] = MagicMock()

# ── 7. Load real business-logic modules ──────────────────────────────────────
sys.path.insert(0, BASE)
_load("app.rag", os.path.join(BASE, "rag.py"))
_load("scripts.ingest", os.path.join(BASE, "ingest.py"))
_load("app.main", os.path.join(BASE, "main.py"))
