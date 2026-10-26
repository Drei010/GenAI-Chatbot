# PDF Knowledge Chatbot

A Streamlit app that answers natural-language questions about an uploaded PDF or Word (.docx) document.
Every supported answer is followed by its **Sources**: the filename, the 1-based page number (or, for a Word
file that doesn't record its page breaks, the section) and a short excerpt copied from the document. When
the document doesn't support an answer, the app says so instead of guessing.

- Ask follow-up questions; the last eight messages give them context.
- Compound questions are split into up to four searches; comparisons come back as tables with a source per
  row, and summaries as cited bullet points.
- Financial answers keep the document's units and periods and keep fiscal and calendar years apart.
- Two models, a primary and a fallback: a rate-limited request is retried once on the fallback.
- Nothing is saved: the document, its search index and the chat live in memory for the browser session.

How it works, and the alternatives considered, are in [docs/architecture.md](docs/architecture.md). The
build log is in [docs/project_journal.md](docs/project_journal.md).

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.13 for you, per
  `.python-version`; Python 3.12 also works)
- API keys for two chat models (they can be from the same provider)
- For the browser tests only: Node.js 20 or newer

## Install

```bash
git clone <this repository>
cd GenAI-Chatbot
uv sync
```

`uv sync` installs the exact versions in `uv.lock`. On first start the app downloads the
`all-MiniLM-L6-v2` embedding model (about 80 MB, ONNX) into Chroma's cache; embeddings are computed
locally after that.

## Provider setup

The app needs a primary and a fallback model. Put the settings in a `.env` file in the project folder
(start from the example), in environment variables, or as top-level keys in `.streamlit/secrets.toml`.
Environment variables win over `.env`, which wins over `secrets.toml`.

```bash
cp .env.example .env
```

| Setting | Required | Meaning |
|---|---|---|
| `PRIMARY_LLM_PROVIDER` | yes | `anthropic`, `openai`, `google_genai`, `groq` or `mistralai` |
| `PRIMARY_LLM_MODEL` | yes | A model ID on that provider, e.g. `claude-opus-5-5` |
| `PRIMARY_LLM_API_KEY` | yes | The provider's API key |
| `FALLBACK_LLM_PROVIDER`, `FALLBACK_LLM_MODEL`, `FALLBACK_LLM_API_KEY` | yes | The same, for the fallback model |
| `LLM_TIMEOUT_SECONDS` | no | Per-call timeout, 1 to 600 (default 120) |
| `LOG_LEVEL` | no | `DEBUG`, `INFO` (default), `WARNING` or `ERROR` |

Example with two providers, so the two rate limits are independent:

```bash
PRIMARY_LLM_PROVIDER=anthropic
PRIMARY_LLM_MODEL=claude-opus-5-5
PRIMARY_LLM_API_KEY=...
FALLBACK_LLM_PROVIDER=openai
FALLBACK_LLM_MODEL=your-openai-model
FALLBACK_LLM_API_KEY=...
```

Notes:

- Models are created with LangChain's `init_chat_model`, so the provider name is LangChain's. The
  `anthropic`, `openai`, `google_genai` and `groq` integrations are installed; for `mistralai` run
  `uv add langchain-mistralai` first. Choose models that support structured output (tool calling or JSON
  schema); the app relies on it for every model call.
- The fallback is used **only** for rate limits (HTTP 429), once per call. A rejected key or model name is
  shown as an error instead, because retrying elsewhere would hide the problem.
- If a setting is missing or still holds the `.env.example` placeholder, the app opens on a setup screen
  that names the setting and how to fix it. Keys are never shown, logged or committed (`.env` and
  `secrets.toml` are git-ignored).
- The app sets no sampling parameters (temperature and so on); some current models accept only their
  defaults.

## Run the app

```bash
uv run streamlit run app.py
```

Open http://localhost:8501, upload a PDF or Word (.docx) file in the sidebar, wait for "is ready", and ask
a question. Use
**Reset chat and document** to start over; uploading a different file replaces the document and its chat.

## Sample document

The demo document is Microsoft's 2023 Annual Report, included as
`samples/microsoft_2023_annual_report.pdf` (the one PDF the repository tracks). It was converted from the
report's Word edition with LibreOffice (`soffice --headless --convert-to pdf 2023_Annual_Report.docx`), so
its page numbers may differ from Microsoft's own PDF on the
[official filing page](https://microsoft.gcs-web.com/static-files/985047d1-e200-4865-837a-c7f03ecb6212);
the evaluation's expected pages refer to the included file. Upload it through the sidebar and try:

- What was Microsoft's total revenue in fiscal year 2023, and how much did it grow?
- And what about operating income? *(a follow-up)*
- Compare revenue for Microsoft's three segments in fiscal years 2023 and 2022 in a table.
- Summarize the key financial results for fiscal year 2023.
- What was Apple's revenue in fiscal year 2023? *(the report doesn't say, so the app declines)*

Indexing its 84 pages takes under half a minute and makes one model call to find the section structure.
The cover and back page are images, so the app notes that two pages have no text to read.

## Word documents

The app also takes Word files (.docx), recognized by their content rather than their name. It reads the
document body in order: paragraphs, tables (one line per row, with cells separated by `|`), text boxes and
the displayed result of fields. Deleted text in tracked changes, hidden text and field codes are skipped.
Paragraphs in Word's heading styles become the sections that sources name.

Word files don't store their page layout, so a source shows a page number only when the file records where
its pages break and those breaks add up: Word saves the page count, and marks where pages broke when it
last laid the document out, and the app checks one against the other. When they don't add up (common for
files written by other apps or converted from another format), each source shows its section and an
excerpt instead, and the sidebar says "No page numbers in this file". The Word edition of the sample report
is one of these: Word saved its page count as 85, but the file marks only 21 of the breaks between those
pages. Exporting such a file to PDF from Word gives page numbers.

Older .doc files and password-protected Word files are turned away with a message that says what to do.

## Tests

```bash
uv run pytest                                   # unit, integration and app tests
uv run pytest --cov                             # the same, with coverage (100% of genai_chatbot/ and app.py)
uv run ruff check . && uv run ruff format --check .
```

The default run is deterministic and offline: test PDFs are generated with reportlab and test Word files
from WordprocessingML snippets (`tests/support/docx.py`), and model calls go to scripted fakes
(`tests/support/fake_llm.py`). No test in it calls a provider or needs a key.

### Browser tests (Playwright)

```bash
npm ci
npx playwright install --with-deps chromium firefox webkit
npx playwright test                      # all three browsers
npx playwright test --project=chromium   # one browser
```

Playwright starts the real app with a rule-based stand-in for the model (`tests/e2e/fake_llm_app.py`) and
uploads generated PDFs and Word files. The tests cover upload, a question with its sources, a follow-up,
bullet and table answers, an unsupported question, an invalid file, reset, replacement, and Word files
cited by page and by section. Screenshots of each state are saved to `test-results/screenshots/<browser>/`.

### Evaluation

`tests/evaluation/annual_report_questions.jsonl` holds 16 questions about the sample report: facts,
financial figures, a comparison table, a summary, a multi-part question, follow-ups and two questions the
report can't answer. Each answerable question lists its known facts, the points a correct answer must
state (with acceptable phrasings), evidence phrases from the report, and the expected source pages.

The question set is checked against the included sample:

```bash
# Check every fact against the report (no model calls); add --write to refresh the expected pages
# after changing a question or the sample
uv run python -m tests.evaluation.locate_pages
uv run pytest -m "evaluation and not live_llm"

# Run the questions against your configured models (costs tokens; never part of CI)
uv run python -m tests.evaluation.run_eval
uv run pytest -m "live_llm and evaluation"     # the same run, as a test
```

The live run uses the app's own pipeline and writes `test-results/evaluation/annual_report.md` (and
`.json`) with each answer, the pages it cited and the scores. It passes when every unsupported question
is declined, at least 85% of answerable questions are answered, 80% of the expected points appear in the
answers, and 80% of answers cite an expected page.

## Continuous integration

`.github/workflows/test.yml` runs on pushes to `main` and on pull requests: ruff and pytest (with a 100%
coverage floor), then Playwright on Chromium, Firefox and WebKit, uploading screenshots and reports as
artifacts. Live-model tests are excluded.

## Privacy and data handling

- The document, its chunks and embeddings, and the chat stay in server memory for the browser session and are
  dropped on reset, replacement, or when the session ends. Nothing is written to disk except the
  embedding model cache.
- Embeddings are computed locally. The question and the retrieved excerpts (up to four passages per
  search) are sent to the configured model provider. For complex documents (long, with tables, or with
  numbered sections), heading candidates (the first 100 characters of up to 400 lines) are also sent once
  at upload to find the section structure.
- Logs record stage names, durations, counts, the provider route, fallback use and outcomes under session
  and request IDs. They never contain keys, document text, prompts, questions or excerpts.

## Project layout

```text
app.py                       Streamlit UI
genai_chatbot/
  config.py                  Settings (Pydantic) from env, .env or Streamlit secrets
  extraction.py              Recognizes an upload as a PDF or Word file and reads it
  documents.py               Upload problems and messages, limits, document IDs
  pdf_processing.py          PDF page text extraction and cleaning
  docx_processing.py         Word text extraction, heading styles and page breaks
  chunking.py                Complexity rules, section detection, chunking and validation
  providers.py               Model factory and the primary/fallback router
  vector_store.py            In-memory Chroma collections with local embeddings
  citations.py               Citation checking, renumbering and excerpts
  prompts.py                 Prompt templates
  schemas.py                 Validated data models
  observability.py           Content-free structured logging
  graphs/ingestion.py        LangGraph: extract, classify, chunk, validate, index
  graphs/qa.py               LangGraph: resolve follow-up, plan, retrieve, answer, check citations
tests/
  unit/, integration/        pytest suites
  evaluation/                Question set, page locator, harness and live runner
  e2e/                       Playwright tests and the stand-in model app
  support/                   Generated PDFs and Word files, and fake models
docs/architecture.md         Design and alternatives considered
docs/project_journal.md      Per-feature build log
```

## Known limitations (v1)

- One document at a time, and the chat lasts only as long as the browser session.
- No OCR: scanned PDFs without a text layer are rejected with an explanation, and text inside pictures in
  Word files isn't read.
- Word files get page numbers only when the page breaks they record add up (see *Word documents*);
  otherwise sources name the section. Headers, footers, footnotes, endnotes and comments in Word files
  aren't read, and older .doc files aren't supported.
- Tables are read as lines of text (pypdf), so complex multi-column tables can lose their column structure.
- Every source cites a single page; a passage that continues across a page break is split at the break.
- Answers appear when they are complete (not streamed), because citations are checked first.
- Excerpts are chosen by word and number overlap, so a paraphrased claim may get a less precise excerpt.
- Retrieval uses embeddings only (no keyword search), so a question worded very differently from the
  document ("cut jobs" for "reduced our workforce") can miss the right passage.
- All sessions share one server process, so memory grows with the number of open sessions and documents.

## Troubleshooting

- **The app opens on a setup screen.** A provider setting is missing or invalid; the screen names it.
  Fix `.env` (or your environment) and restart.
- **"The AI provider rejected the configured API key…"** The key for the named model is wrong or lacks
  access. Questions fail the same way until it's fixed; the document still loads.
- **"The AI service is getting too many requests…"** Both models were rate limited. Wait a minute, or
  configure a fallback on a different provider.
- **"No selectable text was found in this PDF…"** It's a scan or image-only PDF; v1 has no OCR.
- **"This is an older Word (.doc) file…"** Open it in Word and save it as .docx or PDF.
- **Sources show a section instead of a page number.** The Word file doesn't record where its pages
  break. Export it to PDF from Word to get page numbers.
- **The first start is slow.** The embedding model is downloading; later starts use the cache.
