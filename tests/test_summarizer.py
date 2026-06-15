"""Integration tests for src/summarizer.py::summarize_document."""

import pytest

TOP_LEVEL_KEYS = {"doc_type", "filename", "metadata", "summary", "model_used"}
PAPER_METADATA_KEYS = {"title", "authors", "affiliations", "year", "venue"}
PATENT_METADATA_KEYS = {"title", "inventors", "company", "year", "patent_number"}
PAPER_SUMMARY_KEYS = {
    "motivation",
    "problem_statement",
    "novelty",
    "approach",
    "experiments",
    "results",
    "conclusion",
    "limitations",
}
PATENT_SUMMARY_KEYS = {
    "technical_field",
    "independent_claims",
    "novelty_over_prior_art",
    "figures",
}


@pytest.mark.integration
def test_paper_result_has_top_level_keys(paper_summary):
    assert set(paper_summary.keys()) >= TOP_LEVEL_KEYS


@pytest.mark.integration
def test_patent_result_has_top_level_keys(patent_summary):
    assert set(patent_summary.keys()) >= TOP_LEVEL_KEYS


@pytest.mark.integration
def test_paper_metadata_keys(paper_summary):
    assert set(paper_summary["metadata"].keys()) >= PAPER_METADATA_KEYS


@pytest.mark.integration
def test_patent_metadata_keys(patent_summary):
    assert set(patent_summary["metadata"].keys()) >= PATENT_METADATA_KEYS


@pytest.mark.integration
def test_paper_summary_keys(paper_summary):
    assert set(paper_summary["summary"].keys()) >= PAPER_SUMMARY_KEYS


@pytest.mark.integration
def test_patent_summary_keys(patent_summary):
    assert set(patent_summary["summary"].keys()) >= PATENT_SUMMARY_KEYS


@pytest.mark.integration
@pytest.mark.parametrize("section", ["metadata", "summary"])
def test_paper_no_none_or_empty_values(paper_summary, section):
    for key, value in paper_summary[section].items():
        assert value is not None, f"paper {section}[{key!r}] is None"
        assert value != "", f"paper {section}[{key!r}] is empty string"


@pytest.mark.integration
@pytest.mark.parametrize("section", ["metadata", "summary"])
def test_patent_no_none_or_empty_values(patent_summary, section):
    for key, value in patent_summary[section].items():
        assert value is not None, f"patent {section}[{key!r}] is None"
        assert value != "", f"patent {section}[{key!r}] is empty string"


@pytest.mark.integration
def test_paper_no_error_key(paper_summary):
    assert "error" not in paper_summary


@pytest.mark.integration
def test_patent_no_error_key(patent_summary):
    assert "error" not in patent_summary
