"""Summarize a document dict (output of load_document) using the OpenAI API."""

import json

from dotenv import load_dotenv
from openai import OpenAI

from config import MODEL

load_dotenv()

PAPER_PROMPT = """\
You are a research paper analyst. Extract metadata and generate a structured summary from the following paper.

Return ONLY a valid JSON object with exactly this structure. No preamble, no explanation, no markdown backticks:

{{
    "metadata": {{
        "title": "full paper title",
        "authors": ["author 1", "author 2"],
        "affiliations": ["affiliation 1", "affiliation 2"],
        "year": "publication year",
        "venue": "conference or journal name"
    }},
    "summary": {{
        "motivation": "why does this problem matter and what gap does it address",
        "problem_statement": "what specific problem is being solved",
        "novelty": "what is new compared to prior work",
        "approach": "what method or framework is proposed",
        "experiments": "how was the approach validated",
        "results": "key findings with specific numbers where available",
        "conclusion": "main takeaways",
        "limitations": "acknowledged weaknesses or open questions"
    }}
}}

Paper text:
{text}"""

PATENT_PROMPT = """\
You are a patent analyst. Extract metadata and generate a structured summary from the following patent.

Return ONLY a valid JSON object with exactly this structure. No preamble, no explanation, no markdown backticks:

{{
    "metadata": {{
        "title": "full patent title",
        "inventors": ["inventor 1", "inventor 2"],
        "company": "assignee company name",
        "year": "filing or publication year",
        "patent_number": "patent number if present"
    }},
    "summary": {{
        "technical_field": "area of technology this patent covers",
        "independent_claims": "summary of claim 1 and other independent claims — what is the core invention being protected",
        "novelty_over_prior_art": "what this patent claims is new compared to existing technology",
        "figures": "brief description of key figures and what they illustrate"
    }}
}}

Patent text:
{text}"""


def summarize_document(document: dict) -> dict:
    """Summarize a document using the OpenAI API.

    Takes the dictionary returned by ``load_document()`` and sends the full
    text to ``gpt-4o-mini`` with a doc-type-specific prompt. The model is
    instructed to return valid JSON, which is parsed and merged with document
    metadata into the final result.

    Args:
        document: Output of ``load_document()``. Must contain at minimum the
            keys ``doc_type``, ``filename``, and ``text``.

    Returns:
        A dictionary with the following keys:

        - ``doc_type`` (str): ``"paper"`` or ``"patent"``.
        - ``filename`` (str): Original file name.
        - ``metadata`` (dict): Extracted bibliographic metadata.
        - ``summary`` (dict): Structured summary fields.
        - ``model_used`` (str): Model identifier used for the request.

        If the model response cannot be parsed as JSON, ``metadata`` and
        ``summary`` are replaced by a single ``"error"`` key containing the
        raw response text.
    """
    doc_type = document["doc_type"]
    template = PAPER_PROMPT if doc_type == "paper" else PATENT_PROMPT
    prompt = template.format(text=document["text"])

    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        return {
            "doc_type": doc_type,
            "filename": document["filename"],
            "metadata": parsed["metadata"],
            "summary": parsed["summary"],
            "model_used": MODEL,
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "doc_type": doc_type,
            "filename": document["filename"],
            "error": raw,
            "model_used": MODEL,
        }
