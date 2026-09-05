import sys
from pathlib import Path

import pytest

# Make src/ importable without a package install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loader import load_document
from summarizer import summarize_document

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def paper_pdf_path() -> Path:
    """Return the path to the sample research paper PDF."""
    return DATA_DIR / "papers" / "ICST2025.pdf"


@pytest.fixture(scope="session")
def paper_result() -> dict:
    """Load the ICST 2025 paper PDF and return the document dict."""
    path = DATA_DIR / "papers" / "ICST2025.pdf"
    return load_document(str(path))


@pytest.fixture(scope="session")
def patent_result() -> dict:
    """Load the US10942272 patent PDF and return the document dict."""
    path = DATA_DIR / "patents" / "US10942272.pdf"
    return load_document(str(path))


@pytest.fixture(scope="session")
def paper_summary(paper_result) -> dict:
    """Summarize the ICST 2025 paper via the OpenAI API."""
    return summarize_document(paper_result)


@pytest.fixture(scope="session")
def patent_summary(patent_result) -> dict:
    """Summarize the US10942272 patent via the OpenAI API."""
    return summarize_document(patent_result)
