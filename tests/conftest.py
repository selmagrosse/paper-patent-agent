import sys
from pathlib import Path

import pytest

# Make src/ importable without a package install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def paper_pdf_path() -> Path:
    """Return the path to the sample research paper PDF."""
    return DATA_DIR / "papers" / "ICST2025.pdf"
