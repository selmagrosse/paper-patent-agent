"""Tests for src/loader.py::load_document."""

import pytest

from loader import load_document


def test_load_document_returns_expected_keys(paper_pdf_path):
    """load_document result contains all required top-level keys."""
    result = load_document(str(paper_pdf_path))
    assert set(result.keys()) == {"filename", "page_count", "doc_type", "pages", "text"}


def test_load_document_missing_file_raises_file_not_found_error():
    """load_document raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_document("/nonexistent/path/document.pdf")


def test_paper_pdf_metadata(paper_pdf_path):
    """IEEE ICST paper returns page_count > 0, doc_type 'paper', and text > 100 chars."""
    result = load_document(str(paper_pdf_path))
    assert result["page_count"] > 0
    assert result["doc_type"] == "paper"
    assert len(result["text"]) > 100


def test_full_text_preserves_capitalization(paper_pdf_path):
    """The text field retains original casing and contains uppercase characters."""
    result = load_document(str(paper_pdf_path))
    assert any(c.isupper() for c in result["text"])
