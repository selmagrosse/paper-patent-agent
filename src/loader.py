"""Document loader for PDF papers and patents."""

from pathlib import Path

import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pypdf import PdfReader

load_dotenv()

PATENT_KEYWORDS = {"claim", "claims", "patent", "embodiment", "prior art"}

# pypdf pages with fewer characters than this threshold are considered empty
# and trigger OCR as a fallback.
_OCR_THRESHOLD = 50


def _ocr_pdf(file_path: str) -> list[str]:
    """Convert each PDF page to an image and run Tesseract OCR on it.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of extracted text strings, one per page.
    """
    images = convert_from_path(file_path)
    return [pytesseract.image_to_string(img) for img in images]


def load_document(file_path: str) -> dict:
    """Load a PDF document and extract its text and metadata.

    Attempts text extraction with pypdf first. If the result is below a
    minimum character threshold (i.e. the PDF is scanned), falls back to
    OCR via pdf2image + Tesseract.

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

    if len(full_text.strip()) < _OCR_THRESHOLD:
        pages = _ocr_pdf(str(path))
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
