"""
tests/test_rag.py
Unit tests for chunking, metadata parsing, and filter building.
Run with: pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from scripts.ingest import chunk_text, parse_metadata_from_filename, chunk_id
from app.rag import _build_filter


# ─── chunk_text ───────────────────────────────────────────────────────────────

def test_chunk_text_short_text():
    """Short text fits in one chunk."""
    result = chunk_text("Hello world", chunk_size=100, chunk_overlap=20)
    assert result == ["Hello world"]


def test_chunk_text_splits_long_text():
    """Long text is split into multiple chunks."""
    text = "word " * 500  # 2500 chars
    result = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 600  # some tolerance for overlap


def test_chunk_text_no_empty_chunks():
    """No empty strings in output."""
    text = "\n\n".join(["Section content here." * 10] * 5)
    result = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert all(len(c.strip()) > 0 for c in result)


# ─── parse_metadata_from_filename ─────────────────────────────────────────────

def test_metadata_finance_version():
    meta = parse_metadata_from_filename("Banner_Finance_9.3.22_ReleaseNotes.pdf")
    assert meta["banner_module"] == "Finance"
    assert meta["banner_version"] == "9.3.22"


def test_metadata_student():
    meta = parse_metadata_from_filename("Banner_Student_9.39_PatchNotes.pdf")
    assert meta["banner_module"] == "Student"
    assert meta["banner_version"] == "9.39"


def test_metadata_no_match():
    meta = parse_metadata_from_filename("random_document.pdf")
    assert meta["banner_module"] is None
    assert meta["banner_version"] is None


def test_metadata_skips_year():
    """Years like 2024 should not be picked as a version."""
    meta = parse_metadata_from_filename("Banner_HR_2024_Notes.txt")
    assert meta["banner_version"] is None


# ─── chunk_id ─────────────────────────────────────────────────────────────────

def test_chunk_id_deterministic():
    id1 = chunk_id("file.pdf", 1, 0)
    id2 = chunk_id("file.pdf", 1, 0)
    assert id1 == id2


def test_chunk_id_unique():
    assert chunk_id("file.pdf", 1, 0) != chunk_id("file.pdf", 1, 1)
    assert chunk_id("file.pdf", 1, 0) != chunk_id("other.pdf", 1, 0)


# ─── _build_filter ────────────────────────────────────────────────────────────

def test_build_filter_none():
    assert _build_filter(None, None) is None


def test_build_filter_version_only():
    f = _build_filter("9.3.22", None)
    assert f == "banner_version eq '9.3.22'"


def test_build_filter_module_only():
    f = _build_filter(None, "Finance")
    assert f == "banner_module eq 'Finance'"


def test_build_filter_both():
    f = _build_filter("9.3.22", "Finance")
    assert "banner_version eq '9.3.22'" in f
    assert "banner_module eq 'Finance'" in f
    assert " and " in f
