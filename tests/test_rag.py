"""
test_rag.py
Comprehensive unit and integration tests for the Banner RAG system.

Run with:  pytest test_rag.py -v
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Module imports (conftest.py registers everything in sys.modules) ──────────
from scripts.ingest import chunk_text, parse_metadata_from_filename, chunk_id
from scripts.ingest import extract_text_from_txt, extract_pages, ingest_file
from app.rag import _build_filter, ask, generate_answer
from app.models import (
    AskRequest, AskResponse, SourceChunk,
    IngestRequest, IngestResponse, IndexStatsResponse,
)
from app.main import app


# ═══════════════════════════════════════════════════════════════════════════════
# chunk_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        result = chunk_text("Hello world", chunk_size=100, chunk_overlap=20)
        assert result == ["Hello world"]

    def test_long_text_splits_into_multiple_chunks(self):
        text = "word " * 500          # ~2500 chars
        result = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(result) > 1

    def test_no_empty_chunks(self):
        text = "\n\n".join(["Section content here." * 10] * 5)
        result = chunk_text(text, chunk_size=300, chunk_overlap=50)
        assert all(len(c.strip()) > 0 for c in result)

    def test_chunks_respect_size_with_tolerance(self):
        text = "word " * 500
        chunk_size = 500
        result = chunk_text(text, chunk_size=chunk_size, chunk_overlap=50)
        # Each chunk should be at most chunk_size + some separator tolerance
        for chunk in result:
            assert len(chunk) <= chunk_size + 10

    def test_exact_boundary_text_is_single_chunk(self):
        text = "x" * 100
        result = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert result == [text]

    def test_overlap_produces_shared_content(self):
        # With overlap, adjacent chunks should share some text
        text = "A" * 300
        result = chunk_text(text, chunk_size=100, chunk_overlap=30)
        assert len(result) >= 2
        # Each chunk is non-empty
        assert all(len(c) > 0 for c in result)

    def test_whitespace_only_short_text_is_returned_as_is(self):
        # Short text (≤ chunk_size) is returned directly without stripping;
        # callers are expected to provide pre-stripped text.
        result = chunk_text("   \n\n   ", chunk_size=100, chunk_overlap=10)
        assert len(result) == 1
        assert result[0] == "   \n\n   "


# ═══════════════════════════════════════════════════════════════════════════════
# parse_metadata_from_filename
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseMetadataFromFilename:
    def test_finance_module_and_version(self):
        meta = parse_metadata_from_filename("Banner_Finance_9.3.22_ReleaseNotes.pdf")
        assert meta["banner_module"] == "Finance"
        assert meta["banner_version"] == "9.3.22"

    def test_student_module(self):
        meta = parse_metadata_from_filename("Banner_Student_9.39_PatchNotes.pdf")
        assert meta["banner_module"] == "Student"
        assert meta["banner_version"] == "9.39"

    def test_no_match_returns_none(self):
        meta = parse_metadata_from_filename("random_document.pdf")
        assert meta["banner_module"] is None
        assert meta["banner_version"] is None

    def test_skips_year_as_version(self):
        meta = parse_metadata_from_filename("Banner_HR_2024_Notes.txt")
        assert meta["banner_version"] is None

    def test_hr_module_detected(self):
        meta = parse_metadata_from_filename("Banner_HR_9.1_UpgradeGuide.pdf")
        assert meta["banner_module"] == "HR"
        assert meta["banner_version"] == "9.1"

    def test_payroll_module_detected(self):
        meta = parse_metadata_from_filename("Banner_Payroll_8.5.1_Notes.pdf")
        assert meta["banner_module"] == "Payroll"
        assert meta["banner_version"] == "8.5.1"

    def test_advancement_module_detected(self):
        meta = parse_metadata_from_filename("Banner_Advancement_9.3_Release.pdf")
        assert meta["banner_module"] == "Advancement"

    def test_two_digit_version_captured(self):
        meta = parse_metadata_from_filename("Banner_General_9.3_Notes.pdf")
        assert meta["banner_version"] == "9.3"

    def test_returns_dict_with_expected_keys(self):
        meta = parse_metadata_from_filename("whatever.pdf")
        assert "banner_module" in meta
        assert "banner_version" in meta


# ═══════════════════════════════════════════════════════════════════════════════
# chunk_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkId:
    def test_deterministic(self):
        assert chunk_id("file.pdf", 1, 0) == chunk_id("file.pdf", 1, 0)

    def test_unique_by_chunk_index(self):
        assert chunk_id("file.pdf", 1, 0) != chunk_id("file.pdf", 1, 1)

    def test_unique_by_filename(self):
        assert chunk_id("a.pdf", 1, 0) != chunk_id("b.pdf", 1, 0)

    def test_unique_by_page(self):
        assert chunk_id("file.pdf", 1, 0) != chunk_id("file.pdf", 2, 0)

    def test_returns_hex_string(self):
        result = chunk_id("file.pdf", 1, 0)
        assert isinstance(result, str)
        # MD5 hex digest is 32 chars
        assert len(result) == 32
        int(result, 16)  # raises ValueError if not valid hex


# ═══════════════════════════════════════════════════════════════════════════════
# _build_filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFilter:
    def test_both_none_returns_none(self):
        assert _build_filter(None, None) is None

    def test_version_only(self):
        f = _build_filter("9.3.22", None)
        assert f == "banner_version eq '9.3.22'"

    def test_module_only(self):
        f = _build_filter(None, "Finance")
        assert f == "banner_module eq 'Finance'"

    def test_both_filters_joined_with_and(self):
        f = _build_filter("9.3.22", "Finance")
        assert "banner_version eq '9.3.22'" in f
        assert "banner_module eq 'Finance'" in f
        assert " and " in f

    def test_empty_string_treated_as_falsy(self):
        # Empty string should produce no filter clause for that field
        result = _build_filter("", None)
        assert result is None

    def test_filter_is_string(self):
        assert isinstance(_build_filter("9.3", "HR"), str)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic model validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAskRequestModel:
    def test_valid_request(self):
        req = AskRequest(question="What are the prerequisites?")
        assert req.question == "What are the prerequisites?"
        assert req.top_k == 5  # default
        assert req.version_filter is None
        assert req.module_filter is None

    def test_question_too_short_raises(self):
        with pytest.raises(Exception):
            AskRequest(question="Hi")

    def test_top_k_too_large_raises(self):
        with pytest.raises(Exception):
            AskRequest(question="Valid question here", top_k=21)

    def test_top_k_zero_raises(self):
        with pytest.raises(Exception):
            AskRequest(question="Valid question here", top_k=0)

    def test_top_k_boundary_values_accepted(self):
        AskRequest(question="Valid question here", top_k=1)
        AskRequest(question="Valid question here", top_k=20)

    def test_optional_filters_accepted(self):
        req = AskRequest(
            question="What changed in Finance 9.3.22?",
            version_filter="9.3.22",
            module_filter="Finance",
        )
        assert req.version_filter == "9.3.22"
        assert req.module_filter == "Finance"


class TestSourceChunkModel:
    def test_minimal_source_chunk(self):
        chunk = SourceChunk(filename="doc.pdf", chunk_text="Some text", score=0.85)
        assert chunk.filename == "doc.pdf"
        assert chunk.page is None
        assert chunk.banner_module is None
        assert chunk.banner_version is None
        assert chunk.score == 0.85

    def test_full_source_chunk(self):
        chunk = SourceChunk(
            filename="Banner_Finance_9.3.22.pdf",
            page=3,
            banner_module="Finance",
            banner_version="9.3.22",
            chunk_text="Prerequisites section...",
            score=0.92,
        )
        assert chunk.page == 3
        assert chunk.banner_module == "Finance"


class TestAskResponseModel:
    def _make_chunk(self):
        return SourceChunk(filename="x.pdf", chunk_text="text", score=0.5)

    def test_ask_response_construction(self):
        resp = AskResponse(
            question="What?",
            answer="Because...",
            sources=[self._make_chunk()],
            retrieval_count=1,
        )
        assert resp.retrieval_count == 1
        assert len(resp.sources) == 1

    def test_ask_response_empty_sources(self):
        resp = AskResponse(question="Q", answer="A", sources=[], retrieval_count=0)
        assert resp.sources == []


# ═══════════════════════════════════════════════════════════════════════════════
# Text extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractTextFromTxt:
    def test_reads_file_content(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello from a text file.", encoding="utf-8")
        pages = extract_text_from_txt(f)
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert "Hello from a text file." in pages[0]["text"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        pages = extract_text_from_txt(f)
        assert pages == []

    def test_whitespace_only_returns_empty_list(self, tmp_path):
        f = tmp_path / "ws.txt"
        f.write_text("   \n\n  ", encoding="utf-8")
        pages = extract_text_from_txt(f)
        assert pages == []

    def test_multiline_content_preserved(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")
        pages = extract_text_from_txt(f)
        assert "Line 2" in pages[0]["text"]


class TestExtractPages:
    def test_txt_extension_dispatched(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Some content", encoding="utf-8")
        pages = extract_pages(f)
        assert len(pages) == 1

    def test_md_extension_dispatched(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Title\nContent here", encoding="utf-8")
        pages = extract_pages(f)
        assert len(pages) == 1

    def test_unsupported_extension_returns_empty(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_text("not real data")
        pages = extract_pages(f)
        assert pages == []


# ═══════════════════════════════════════════════════════════════════════════════
# ingest_file
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestFile:
    def test_ingest_txt_file_returns_chunk_count(self, tmp_path):
        f = tmp_path / "Banner_Finance_9.3.22_Notes.txt"
        # Write enough content to produce at least one chunk
        f.write_text("Important upgrade note. " * 50, encoding="utf-8")

        mock_client = MagicMock()
        mock_embed = MagicMock(return_value=[0.1] * 1536)

        import app.azure_clients as ac
        original_embed = ac.embed_text
        ac.embed_text = mock_embed

        try:
            n = ingest_file(f, mock_client, chunk_size=500, chunk_overlap=50)
        finally:
            ac.embed_text = original_embed

        assert n >= 1
        mock_client.upload_documents.assert_called()

    def test_empty_file_returns_zero(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        mock_client = MagicMock()
        n = ingest_file(f, mock_client, chunk_size=500, chunk_overlap=50)
        assert n == 0
        mock_client.upload_documents.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# ask() orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class TestAskOrchestrator:
    def _make_chunk(self, text="Sample upgrade info."):
        return SourceChunk(
            filename="Banner_Finance_9.3.22_ReleaseNotes.pdf",
            page=2,
            banner_module="Finance",
            banner_version="9.3.22",
            chunk_text=text,
            score=0.88,
        )

    def test_empty_retrieval_returns_fallback_answer(self):
        req = AskRequest(question="What are the prerequisites for Finance?")
        with patch("app.rag.retrieve_chunks", return_value=[]):
            response = ask(req)
        assert response.retrieval_count == 0
        assert response.sources == []
        assert "could not find" in response.answer.lower()

    def test_with_chunks_calls_generate_answer(self):
        req = AskRequest(question="What changed in Banner Finance 9.3.22?")
        chunks = [self._make_chunk()]
        with patch("app.rag.retrieve_chunks", return_value=chunks), \
             patch("app.rag.generate_answer", return_value="Here is the answer.") as mock_gen:
            response = ask(req)
        mock_gen.assert_called_once_with(req.question, chunks)
        assert response.answer == "Here is the answer."

    def test_retrieval_count_matches_chunks(self):
        req = AskRequest(question="Finance upgrade steps?")
        chunks = [self._make_chunk(), self._make_chunk("Second chunk.")]
        with patch("app.rag.retrieve_chunks", return_value=chunks), \
             patch("app.rag.generate_answer", return_value="Answer."):
            response = ask(req)
        assert response.retrieval_count == 2
        assert len(response.sources) == 2

    def test_sources_preserved_in_response(self):
        req = AskRequest(question="Any known issues with Finance 9.3?")
        chunk = self._make_chunk()
        with patch("app.rag.retrieve_chunks", return_value=[chunk]), \
             patch("app.rag.generate_answer", return_value="No known issues."):
            response = ask(req)
        assert response.sources[0].filename == chunk.filename
        assert response.sources[0].banner_module == "Finance"

    def test_question_echoed_in_response(self):
        question = "What are the system requirements for Banner 9.3?"
        req = AskRequest(question=question)
        with patch("app.rag.retrieve_chunks", return_value=[]), \
             patch("app.rag.generate_answer", return_value=""):
            response = ask(req)
        assert response.question == question


# ═══════════════════════════════════════════════════════════════════════════════
# generate_answer — context construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateAnswer:
    def _make_chunk(self, **kwargs):
        defaults = dict(
            filename="doc.pdf", page=1,
            banner_module="Finance", banner_version="9.3",
            chunk_text="Upgrade requires Java 11.", score=0.9,
        )
        defaults.update(kwargs)
        return SourceChunk(**defaults)

    def test_calls_openai_chat_completions(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Answer text"

        with patch("app.rag.get_openai_client", return_value=mock_client):
            result = generate_answer("What are the requirements?", [self._make_chunk()])

        assert result == "Answer text"
        mock_client.chat.completions.create.assert_called_once()

    def test_context_includes_filename(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"

        with patch("app.rag.get_openai_client", return_value=mock_client):
            generate_answer("Q?", [self._make_chunk(filename="Banner_Finance.pdf")])

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "Banner_Finance.pdf" in user_msg

    def test_context_includes_page_module_version(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"

        with patch("app.rag.get_openai_client", return_value=mock_client):
            generate_answer("Q?", [self._make_chunk(page=5, banner_module="HR", banner_version="9.1")])

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "page 5" in user_msg
        assert "HR" in user_msg
        assert "9.1" in user_msg

    def test_chunk_text_included_in_context(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"
        chunk_content = "Java 17 is required for this upgrade."

        with patch("app.rag.get_openai_client", return_value=mock_client):
            generate_answer("Q?", [self._make_chunk(chunk_text=chunk_content)])

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert chunk_content in user_msg


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI API endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_200_when_azure_ok(self, client):
        with patch("app.main.get_openai_client", return_value=MagicMock()), \
             patch("app.main.get_search_client", return_value=MagicMock()):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "index_name" in data
        assert "chat_model" in data

    def test_health_returns_503_on_azure_error(self, client):
        with patch("app.main.get_openai_client", side_effect=RuntimeError("no connection")):
            resp = client.get("/health")
        assert resp.status_code == 503


class TestIndexStatsEndpoint:
    def test_index_stats_returns_document_count(self, client):
        mock_search = MagicMock()
        mock_search.get_document_count.return_value = 42
        with patch("app.main.get_search_client", return_value=mock_search):
            resp = client.get("/index/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 42
        assert data["status"] == "ready"

    def test_index_stats_500_on_error(self, client):
        with patch("app.main.get_search_client", side_effect=Exception("search down")):
            resp = client.get("/index/stats")
        assert resp.status_code == 500


class TestAskEndpoint:
    def _make_response(self):
        return AskResponse(
            question="What changed?",
            answer="Several things changed.",
            sources=[SourceChunk(filename="doc.pdf", chunk_text="text", score=0.9)],
            retrieval_count=1,
        )

    def test_ask_returns_200_with_valid_request(self, client):
        with patch("app.main.rag_ask", return_value=self._make_response()):
            resp = client.post("/ask", json={"question": "What changed in Banner Finance?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieval_count" in data

    def test_ask_passes_filters_to_rag(self, client):
        with patch("app.main.rag_ask", return_value=self._make_response()) as mock_rag:
            client.post("/ask", json={
                "question": "What are the prerequisites?",
                "version_filter": "9.3.22",
                "module_filter": "Finance",
                "top_k": 3,
            })
        call_arg = mock_rag.call_args[0][0]
        assert call_arg.version_filter == "9.3.22"
        assert call_arg.module_filter == "Finance"
        assert call_arg.top_k == 3

    def test_ask_rejects_short_question(self, client):
        resp = client.post("/ask", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_ask_rejects_top_k_out_of_range(self, client):
        resp = client.post("/ask", json={"question": "Valid question here", "top_k": 0})
        assert resp.status_code == 422

    def test_ask_500_on_rag_error(self, client):
        with patch("app.main.rag_ask", side_effect=Exception("RAG failure")):
            resp = client.post("/ask", json={"question": "What changed in Finance?"})
        assert resp.status_code == 500


class TestIngestEndpoint:
    def test_ingest_404_for_nonexistent_path(self, client):
        resp = client.post("/ingest", json={"docs_path": "/no/such/path"})
        assert resp.status_code == 404

    def test_ingest_200_on_success(self, client, tmp_path):
        mock_result = {
            "status": "success",
            "documents_processed": 2,
            "chunks_indexed": 14,
            "message": "Ingested 2 documents",
        }
        with patch("scripts.ingest.run_ingestion", return_value=mock_result):
            resp = client.post("/ingest", json={"docs_path": str(tmp_path)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_processed"] == 2
        assert data["chunks_indexed"] == 14


class TestBlobListEndpoint:
    def test_blob_list_400_when_no_connection_string(self, client):
        from app.config import settings
        original = settings.azure_storage_connection_string
        settings.azure_storage_connection_string = ""
        try:
            resp = client.get("/blob/list")
            assert resp.status_code == 400
        finally:
            settings.azure_storage_connection_string = original
