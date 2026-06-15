"""Entry point: load a PDF and print a structured summary."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loader import load_document
from summarizer import summarize_document

DEFAULT_PDF = Path(__file__).parent / "data" / "papers" / "ICST2025.pdf"


def main(pdf_path: Path = DEFAULT_PDF) -> None:
    """Run the load → summarize pipeline and print the result as JSON.

    Args:
        pdf_path: Path to the PDF file to process. Defaults to the ICST 2025
            paper in data/papers/.
    """
    document = load_document(str(pdf_path))
    result = summarize_document(document)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        from agent import PaperPatentAgent
        agent = PaperPatentAgent()
        response = agent.run("Find me the paper that introduced transformer architecture")
        print(response)
    else:
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
        main(path)
