# Paper & Patent Agent

A Python research assistant that searches arXiv and Google Patents, downloads PDFs into a local library, and summarizes documents using GPT-4o-mini. An LLM-powered validator agent judges whether each search result matches the original request before asking for confirmation.

---

## Project structure

```
paper-patent-agent/
├── main.py                  # Entry point — summariser pipeline or agent mode
├── requirements.txt
├── pytest.ini
├── src/
│   ├── base_agent.py        # BaseAgent: memory, tool registry, LLM
│   ├── agent.py             # PaperPatentAgent: search, download, library
│   ├── validator_agent.py   # ValidatorAgent: judges search result relevance
│   ├── loader.py            # PDF loader with OCR fallback for scanned files
│   └── summarizer.py        # Structured summarisation via OpenAI
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_loader.py       # Unit tests for loader
│   └── test_summarizer.py   # Integration tests for summarizer (calls API)
└── data/
    ├── papers/              # Downloaded / manually added research papers
    └── patents/             # Downloaded / manually added patents
```

---

## Features

| Component | What it does |
|---|---|
| `loader.py` | Extracts text from PDFs using `pypdf`; falls back to Tesseract OCR for scanned files |
| `summarizer.py` | Sends full document text to GPT-4o-mini with a paper or patent prompt; returns structured JSON |
| `PaperPatentAgent` | Classifies user intent, runs `search_arxiv` or `search_patents_google`, validates the result, asks for confirmation, downloads the PDF |
| `ValidatorAgent` | Rates whether a candidate result satisfies the user request (confidence 0–1 + reasoning) |

---

## Requirements

### System dependencies

```bash
brew install tesseract poppler
```

Tesseract is required for OCR on scanned PDFs. Poppler provides `pdfinfo`, used by `pdf2image`.

### Python dependencies

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

### API keys

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
SERPAPI_KEY=...
```

`SERPAPI_KEY` is only needed for patent search (`search_patents_google`). A key can be obtained at [serpapi.com](https://serpapi.com).

---

## How to run

### Summarise a PDF

Runs the load → summarise pipeline on a single PDF and prints structured JSON.

The default test paper is **Vaswani et al. (2017) — "Attention Is All You Need"**. Download it from arXiv and place it in `data/papers/` before running:

```bash
mkdir -p data/papers
curl -L https://arxiv.org/pdf/1706.03762 -o data/papers/vaswani2017.pdf
```

Then run:

```bash
# Default: data/papers/vaswani2017.pdf
venv/bin/python main.py

# Custom path
venv/bin/python main.py data/papers/vaswani2017.pdf
```

### Run the agent

Launches `PaperPatentAgent` in interactive mode. The agent auto-loads all PDFs from `data/papers/` and `data/patents/` at startup.

```bash
venv/bin/python main.py agent
```

Example requests the agent handles:

- `"Find me the paper that introduced the transformer architecture"`
- `"Find a paper by Vaswani about attention mechanisms"`
- `"Find a US patent about LiDAR sensor fusion for autonomous driving"`
- `"Find a patent about autonomous vehicle route planning filed after 2020"`

After each search the agent shows the result, the validator's confidence score, and asks whether to download it into the library.

---

## Running tests

```bash
# All tests
venv/bin/python -m pytest tests/ -v

# Unit tests only (no API calls)
venv/bin/python -m pytest tests/ -v -m "not integration"

# Integration tests only (calls OpenAI API)
venv/bin/python -m pytest tests/ -v -m integration
```

---

## How the agent pipeline works

```
User request
    │
    ▼
classify_intent_and_extract_filters()   ← one LLM call → { tool, params }
    │
    ▼
execute_tool()                          ← search_arxiv or search_patents_google
    │   fetches top 5 candidates
    │   llm_pick_best() selects the most relevant one
    │
    ▼
validator.validate()                    ← one LLM call → { valid, confidence, reasoning }
    │
    ▼
ask_confirmation()                      ← shows result + validator verdict to user
    │
    ├── yes → download_and_load()       ← downloads PDF, runs OCR if needed, appends to library
    └── no  → cancelled
```
