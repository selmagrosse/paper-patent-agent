"""Entry point: load a PDF and print a structured summary."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loader import load_document
from summarizer import summarize_document

DEFAULT_PDF = Path(__file__).parent / "data" / "papers" / "vaswani2017.pdf"


def main(pdf_path: Path = DEFAULT_PDF) -> None:
    """Run the load → summarize pipeline and print the result as JSON.

    Args:
        pdf_path: Path to the PDF file to process. Defaults to the Vaswani 2017
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
        # response = agent.run("Find me a paper by Vaswani about attention mechanisms")
        # print(response)
        # response = agent.run("Find me a paper about SOTIF from 2025")
        # print(response)
        # response = agent.run("Find me a patent about LiDAR sensor fusion for autonomous driving")
        # print(response)
        # response = agent.run("Find me a US patent about LiDAR sensor fusion for autonomous driving")
        # print(response)
    else:
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
        main(path)
