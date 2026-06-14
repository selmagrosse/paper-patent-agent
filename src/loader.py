"""Document loader for PDF papers and patents."""

from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

PATENT_KEYWORDS = {"claim", "claims", "patent", "embodiment", "prior art"}

def load_document(file_path: str) -> dict:
    """Load a PDF document and extract its text and metadata.

    Reads each page of the PDF, detects whether the document is a
    research paper or a patent based on the presence of patent-specific
    keywords, and returns a structured result.

    Args:
        file_path: Absolute or relative path to a PDF file.

    Returns:
        A dictionary with the following keys:
            - ``filename`` (str): Base name of the file.
            - ``page_count`` (int): Total number of pages.
            - ``doc_type`` (str): ``"patent"`` or ``"paper"``.
            - ``pages`` (list[str]): Extracted text for each page, original casing.
            - ``text`` (str): Full concatenated text, original casing.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        pypdf.errors.PdfReadError: If the file is not a valid PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at: {file_path}")

    reader = PdfReader(str(path))

    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)

    # Lowercase only for keyword matching; full_text retains original casing
    text_for_detection = full_text.lower()
    doc_type = "patent" if any(kw in text_for_detection for kw in PATENT_KEYWORDS) else "paper"

    return {
        "filename": path.name,
        "page_count": len(pages),
        "doc_type": doc_type,
        "pages": pages,
        "text": full_text,
    }
