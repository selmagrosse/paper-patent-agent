"""PaperPatentAgent — search, download, and query research papers and patents."""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import arxiv
import requests
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch

from base_agent import BaseAgent
from loader import load_document
from validator_agent import ValidatorAgent

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL = "gpt-4o-mini"

def _extract_jurisdiction(patent_id: str) -> str:
    clean = patent_id.removeprefix("patent/").split("/")[0]
    match = re.match(r"^([A-Z]{2})", clean)
    return match.group(1) if match else ""


def _resolve_pdf_url(raw_pdf: str, patent_id: str) -> str:
    if raw_pdf:
        return raw_pdf
    clean_id = patent_id.removeprefix("patent/").removesuffix("/en")
    return f"https://patents.google.com/patent/{clean_id}"


INTENT_PROMPT = """\
You are an intent classifier for a research assistant that searches papers and patents.

Given the user request below, return ONLY a valid JSON object with exactly this structure:
{{
    "tool": "<one of: search_arxiv, search_patents_google>",
    "params": {{
        "query": "<core search query>",
        ... any optional filters that apply ...
    }}
}}

Optional params for search_arxiv: author, year, category, title
Optional params for search_patents_google: assignee, inventor, after_year, before_year, patent_type, jurisdiction (two-letter country code, e.g. 'US', 'EP', 'JP', 'CN')

User request: {user_request}"""


class PaperPatentAgent(BaseAgent):
    """Agent that searches arXiv and Google Patents, downloads PDFs, and answers
    queries over a local document library.

    Attributes:
        library: List of document dicts returned by :func:`load_document`.
            Documents are appended each time :meth:`download_and_load` is called.
        validator: A :class:`ValidatorAgent` instance used to double-check
            whether search candidates genuinely satisfy a user request.
    """

    def __init__(self) -> None:
        super().__init__(model_name=MODEL)
        self.library: list[dict] = []
        self._client = OpenAI()
        self.validator = ValidatorAgent()

        self.register_tool("search_arxiv", self.search_arxiv)
        self.register_tool("search_patents_google", self.search_patents_google)
        self.register_tool("download_and_load", self.download_and_load)

        # Auto-load existing documents from data folders
        self._load_existing_documents()

    def _load_existing_documents(self) -> None:
        """Load all existing PDFs from data/papers/ and data/patents/ into library."""
        for folder in [DATA_DIR / "papers", DATA_DIR / "patents"]:
            if folder.exists():
                for pdf_path in folder.glob("*.pdf"):
                    try:
                        doc = load_document(str(pdf_path))
                        self.library.append(doc)
                        print(f"Loaded: {doc['filename']}")
                    except Exception as e:
                        print(f"Failed to load {pdf_path.name}: {e}")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def llm_pick_best(self, candidates: list[dict], user_request: str) -> int:
        """Use gpt-4o-mini to select the best matching result from a candidate list.

        Generic across papers and patents — requires each candidate to have at
        least ``title`` and ``abstract`` keys.

        Args:
            candidates: List of result dicts, each containing at least
                ``title`` and ``abstract``.
            user_request: The original natural-language request from the user,
                used by the model to judge relevance.

        Returns:
            Integer index (0-based) of the best candidate. Falls back to 0 if
            the model returns an invalid or unparseable index.
        """
        numbered = "\n\n".join(
            f"[{i}] Title: {c['title']}\nAbstract: {c.get('abstract', '')[:300]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"A user asked: \"{user_request}\"\n\n"
            f"Here are {len(candidates)} candidate results:\n\n{numbered}\n\n"
            "Return ONLY a valid JSON object with exactly this structure:\n"
            "{\"best_index\": <integer 0 to "
            f"{len(candidates) - 1}>}}"
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            index = int(parsed["best_index"])
            if 0 <= index < len(candidates):
                return index
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return 0

    def search_arxiv(
        self,
        query: str,
        author: str | None = None,
        year: str | None = None,
        category: str | None = None,
        title: str | None = None,
    ) -> dict:
        """Search arXiv for the best matching paper among the top 5 results.

        Constructs an arXiv query string from *query* and optional filters,
        fetches up to 5 candidates, uses :meth:`llm_pick_best` to select the
        most relevant one, and returns its metadata.

        Args:
            query: Core search terms.
            author: Filter by author name (partial match accepted).
            year: Include year in the search query string.
            category: arXiv category code (e.g. ``"cs.SE"``).
            title: Additional title keywords.

        Returns:
            Dict with keys: ``title``, ``authors``, ``year``, ``abstract``,
            ``arxiv_id``, ``pdf_url``. Returns an ``error`` key on failure.
        """
        if title:
            search_query = f'ti:"{title}"'
        elif author:
            search_query = f'au:"{author}" AND {query}'
        else:
            parts = [query]
            if year:
                parts.append(str(year))
            if category:
                parts.append(f"cat:{category}")
            search_query = " AND ".join(parts)
        client = arxiv.Client()
        search = arxiv.Search(query=search_query, max_results=5)
        results = list(client.results(search))

        if not results:
            return {"error": "No arXiv results found."}

        candidates = [
            {
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "year": str(r.published.year) if r.published else "",
                "abstract": r.summary,
                "arxiv_id": r.entry_id,
                "pdf_url": r.pdf_url,
            }
            for r in results
        ]

        best = self.llm_pick_best(candidates, query)
        return candidates[best]

    def search_patents_google(
        self,
        query: str,
        assignee: str | None = None,
        inventor: str | None = None,
        after_year: str | None = None,
        before_year: str | None = None,
        patent_type: str | None = None,
        jurisdiction: str | None = None,
    ) -> dict:
        """Search Google Patents via SerpAPI.

        Args:
            query: Core search terms.
            assignee: Filter by assignee/company name.
            inventor: Filter by inventor name.
            after_year: Return patents filed after this year (inclusive).
            before_year: Return patents filed before this year (inclusive).
            patent_type: ``"patent"`` or ``"application"``.
            jurisdiction: Two-letter country code to filter by (e.g. ``"US"``,
                ``"EP"``, ``"JP"``, ``"CN"``).

        Returns:
            Dict with keys: ``title``, ``patent_id``, ``assignee``,
            ``inventor``, ``filing_date``, ``abstract``, ``pdf_url``,
            ``jurisdiction``. Returns an ``error`` key on failure.
        """
        params: dict = {
            "engine": "google_patents",
            "q": query,
            "api_key": os.environ.get("SERPAPI_KEY", ""),
        }
        if assignee:
            params["assignee"] = assignee
        if inventor:
            params["inventor"] = inventor
        if after_year:
            params["after"] = f"filing:{after_year}0101"
        if before_year:
            params["before"] = f"filing:{before_year}1231"
        if patent_type:
            params["type"] = patent_type
        if jurisdiction:
            params["country"] = jurisdiction

        search = GoogleSearch(params)
        data = search.get_dict()

        organic = data.get("organic_results", [])
        if not organic:
            return {"error": "No Google Patents results found."}

        candidates = [
            {
                "title": r.get("title", ""),
                "patent_id": r.get("patent_id", ""),
                "assignee": r.get("assignee", ""),
                "inventor": r.get("inventor", ""),
                "filing_date": r.get("filing_date", ""),
                "abstract": r.get("snippet", ""),
                "pdf_url": _resolve_pdf_url(r.get("pdf", ""), r.get("patent_id", "")),
                "jurisdiction": _extract_jurisdiction(r.get("patent_id", "")),
            }
            for r in organic[:5]
        ]

        best = self.llm_pick_best(candidates, query)
        top = candidates[best]
        return candidates[best]

    def download_and_load(self, url: str, source: str = "paper") -> dict:
        """Download a PDF from *url*, save it locally, and add it to the library.

        Files are saved to ``data/papers/`` for papers and ``data/patents/``
        for patents.

        Args:
            url: Direct URL to the PDF file.
            source: ``"paper"`` or ``"patent"`` — determines the save directory.

        Returns:
            The document dict returned by :func:`load_document`, which is also
            appended to :attr:`library`.
        """
        folder = DATA_DIR / ("papers" if source == "paper" else "patents")
        folder.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        dest = folder / filename

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        if not response.content.startswith(b"%PDF"):
            return {"error": "URL did not return a valid PDF"}

        dest.write_bytes(response.content)

        document = load_document(str(dest))
        self.library.append(document)
        return document

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent_and_extract_filters(self, user_request: str) -> dict:
        """Use gpt-4o-mini to classify the user request and extract search params.

        Makes a single LLM call instructing the model to return a JSON object
        with ``tool`` and ``params`` fields. Falls back to a default
        ``search_arxiv`` intent if parsing fails.

        Args:
            user_request: Raw natural-language request from the user.

        Returns:
            Dict with keys ``tool`` (str) and ``params`` (dict).
        """
        prompt = INTENT_PROMPT.format(user_request=user_request)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"tool": "search_arxiv", "params": {"query": user_request}}

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def ask_confirmation(self, result: dict, verdict: dict = None) -> bool:
        """Format *result* and prompt the user to confirm adding it to the library.

        Args:
            result: A search result dict from :meth:`search_arxiv` or
                :meth:`search_patents_google`.
            verdict: Optional validator judgment dict from
                :meth:`ValidatorAgent.validate`, with ``confidence`` and
                ``reasoning`` keys. If provided, surfaced as an extra line
                before the confirmation prompt.

        Returns:
            ``True`` if the user confirms, ``False`` otherwise.
        """
        lines = ["\n--- Search Result ---"]
        for key, value in result.items():
            if key == "abstract":
                value = value[:300] + "..." if len(value) > 300 else value
            lines.append(f"{key.capitalize()}: {value}")
        if verdict is not None:
            lines.append(
                f"Validator confidence: {verdict.get('confidence')} — {verdict.get('reasoning')}"
            )
        lines.append("---------------------")
        print("\n".join(lines))

        answer = input("\nWould you like to add this to your library? (yes/no): ").strip().lower()
        return answer in {"yes", "y"}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, user_request: str) -> str:
        """Process a user request through the full agent pipeline.

        Steps:
        1. Add the request to memory.
        2. Classify intent and extract filters via LLM.
        3. Execute the appropriate tool.
        4. Validate the result, then ask confirmation, then download if confirmed.
        5. Add the final response to memory.

        Args:
            user_request: Natural-language request from the user.

        Returns:
            A human-readable response string.
        """
        self.add_to_memory("user", user_request)

        intent = self.classify_intent_and_extract_filters(user_request)
        tool = intent.get("tool", "search_arxiv")
        params = intent.get("params", {"query": user_request})

        result = self.execute_tool(tool, **params)

        if isinstance(result, dict) and "error" in result:
            response = f"Search failed: {result['error']}"
            self.add_to_memory("agent", response)
            return response

        verdict = self.validator.validate(user_request, result)
        confirmed = self.ask_confirmation(result, verdict)
        if confirmed:
            source = "paper" if tool == "search_arxiv" else "patent"
            pdf_url = result.get("pdf_url", "")
            if not pdf_url:
                response = "No PDF URL available for download."
            else:
                self.execute_tool("download_and_load", url=pdf_url, source=source)
                response = (
                    f"Added '{result.get('title', pdf_url)}' to your library. "
                    f"Library now contains {len(self.library)} document(s)."
                )
        else:
            response = "Cancelled — document not added to library."

        self.add_to_memory("agent", response)
        return response
