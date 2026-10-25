# Project journal

A running log of the build, one entry per feature: what was decided, what went wrong and how it was fixed,
how the prompts changed, the test results, and the tradeoffs accepted. Newest entries go at the bottom.

## Environment notes (2026-10-25)

- Built in a cloud container with Python 3.13, uv 0.11 and Node 22. The network policy allowed PyPI, npm and
  Chroma's model bucket, but blocked Hugging Face, the Playwright browser CDN and both Microsoft download
  hosts for the sample report.
- Consequences: embeddings use Chroma's ONNX build of `all-MiniLM-L6-v2` (downloaded from Chroma's S3 bucket,
  no PyTorch); local browser runs use the preinstalled Chromium only, so Firefox and WebKit run in CI; the
  Microsoft report has to be added by hand (see Feature 7).
- The plan asks for the Ponytail skill for code work. It wasn't available in this environment, so its intent
  (small, test-backed, one-feature-at-a-time changes) is written into `AGENTS.md` instead.

## Feature 1 — Project foundation

**Built:** `pyproject.toml` + `uv.lock`, package skeleton, Pydantic settings (`genai_chatbot/config.py`),
content-free structured logging (`genai_chatbot/observability.py`), the Streamlit shell (`app.py`) with a setup
screen, `.env.example`, a private-safe `.gitignore`, `.streamlit/config.toml`, `AGENTS.md`, and pytest/ruff
configuration.

**Design decisions**

- *Settings:* `pydantic-settings` with sources in this order: keyword arguments, environment variables,
  `.env`, then a custom `StreamlitSecretsSource` for `.streamlit/secrets.toml`. Environment wins over secrets
  so a deployment can override a checked-in-looking secrets file without editing it.
- *Primary and fallback are both required*, each with its own provider, model and key, as the plan asks.
  Using the same provider twice with a different model is allowed; the README recommends a second provider so
  the rate limits are independent.
- *Validation that helps people fix things:* provider names are normalized and checked against the providers
  `init_chat_model` can build (anthropic, openai, google_genai, groq, mistralai), the integration package must
  be importable (the message says which `uv add` to run), blank values count as missing, and keys that still
  hold the `.env.example` placeholder are rejected.
- *No secret ever reaches a message:* setup issues are built from `exc.errors(include_input=False)` and only
  name the variable, never the value, because `str(ValidationError)` prints the rejected input, which would
  include an API key.
- *Logging allowlist:* `log_event` and `log_stage` drop any key that isn't in `SAFE_FIELDS` and any value that
  isn't a scalar. Leaking a question or excerpt then needs an explicit code change rather than a careless
  keyword argument. `pypdf`, `httpx` and `chromadb` loggers are set to ERROR.
- *Streamlit config:* uploads capped at 50 MB, usage stats off, and `showErrorDetails = "none"` so users
  never see a traceback.
- *No sampling parameters:* the app never sets `temperature`/`top_p`/`top_k`. Current Claude models only
  accept the defaults, the 2026-10-21 Anthropic API version rejects the fields outright, and some OpenAI
  reasoning models reject non-default temperatures too. Grounding comes from prompts and validation instead.

**Challenges and fixes**

- The example `.env` block in the setup message rendered as an empty box when it was a fenced block nested in
  a numbered list. Moved it out of the list; confirmed with a Chromium screenshot.
- `config.AppSettings` would happily read a developer's real `.env` or `secrets.toml` during tests. An autouse
  fixture clears the provider variables, disables `.env` loading and stubs the secrets reader for every test.
- `configure_logging` turns off propagation, which hid records from `caplog` in later tests; the test restores
  the logger afterwards.

**Tests:** `uv run pytest --cov=genai_chatbot` → 39 passed, 100% line and branch coverage of `config.py` and
`observability.py`. `ruff check` and `ruff format --check` clean.

**Gate**

- App starts: `streamlit run app.py` answers `/_stcore/health`, and Streamlit `AppTest` runs the script with
  no exception in both the configured and unconfigured state.
- Missing or invalid settings: the setup screen lists each variable with what's wrong and how to fix it
  (tests cover missing, blank, placeholder, unsupported provider, missing integration package, bad timeout
  and log level).
- No secrets in tracked files: hygiene tests confirm `git check-ignore` covers `.env*`, `secrets.toml`,
  uploads, PDFs other than the sample, Chroma folders and test artifacts, and scan every tracked or
  trackable file for API-key patterns.

**Tradeoffs:** requiring a fallback means two keys even for a quick local trial; the setup screen explains
it, and the same key can be reused with a second model.

## Feature 2 — PDF upload and extraction

**Built:** `genai_chatbot/pdf_processing.py` (validation, extraction, cleaning), page and document models in
`genai_chatbot/schemas.py`, the sidebar upload flow in `app.py`, and a reportlab-based PDF factory for tests
(`tests/support/pdfs.py`).

**Design decisions**

- *pypdf* for extraction: pure Python, permissively licensed, and good enough on text-layer PDFs. PyMuPDF
  extracts faster and with better layout but is AGPL; pdfplumber is slower on a 100-page report.
- *Checks in order of cheapness:* empty bytes → `%PDF-` header within the first 1 KB (catches renamed text,
  PNG and Office files before parsing) → parse (any pypdf failure becomes "corrupt") → password (an empty
  user password is tried, so owner-locked PDFs still open) → zero pages → page limit (500) → text layer.
- *Scanned detection:* fewer than 20 letters and digits in the whole document means there's no usable
  text layer. The message says OCR isn't part of v1. Documents with some blank pages still load, and the app
  reports how many pages had no text.
- *One bad page doesn't sink a document:* a page whose content stream throws is kept as an empty page so the
  remaining page numbers stay correct. Only when every page fails is the file reported as corrupt.
- *Cleaning keeps the words:* NFKC (ligatures like "ﬁ" become "fi"), horizontal whitespace collapsed,
  words hyphenated across a line break are rejoined only between lowercase letters (so "2022-\n2023" stays),
  and runs of blank lines squashed. Excerpts shown to users come from this cleaned text, so they're still
  exact copies of what was indexed.
- *Page furniture:* page-number lines and short lines repeated at the top or bottom of at least 60% of pages
  (digits ignored, so "Page 3 of 9" matches "Page 4 of 9") are removed from page edges. Only two lines per
  edge are considered, documents under four pages are left alone, and a page is never stripped to nothing.
- *IDs:* the document ID is the SHA-256 of the uploaded bytes, so re-uploading the same file is recognized
  and collection names never contain the filename.
- *Upload state:* the app keys work on Streamlit's `file_id`, so reruns don't re-process the same upload, a
  different file is a replacement, removing the file clears the document, and "Reset" bumps a generation
  counter that also gives the uploader a fresh key (emptying the widget).

**Challenges and fixes**

- First version of header removal also deleted real content: synthetic pages that all opened with
  "Section N overview" lost that line, and a page made only of repeated lines became empty and was flagged as
  scanned. Added the 80-character limit, the two-line edge window, and a fallback that only strips page
  numbers when everything else would go.
- `PasswordType` moved out of `pypdf.constants` in pypdf 6; imported from the package root instead.
- Upload failures were logged as `PDFProcessingError`; the log now records the specific kind
  (`not_pdf`, `no_text`, …) with outcome `rejected`, still without filename or content.
- reportlab's base fonts can't draw the "ﬁ" ligature, so ligature cleanup is tested on strings rather than
  through a generated PDF.

**Tests:** 69 passed (26 new extraction tests, 4 new AppTest upload tests); coverage 100% of `genai_chatbot/`.

**Gate**

- Pytest covers valid extraction (page numbering, filename, IDs, blank pages, owner-locked PDFs, cleaning,
  header/footer removal) and each failure case: empty file, non-PDF content with a `.pdf` name (text, PNG,
  ZIP), corrupt (truncated, garbled, header-only), zero pages, image-only scan, password-protected, page
  limit, and unreadable pages.
- Upload through the app: verified in Chromium against the running app (screenshots of the ready and
  not-a-PDF states) and with Streamlit AppTest. The Microsoft annual report itself is checked in Feature 7,
  once the file is available.

**Tradeoffs:** pypdf flattens table columns into space-separated lines, so tables arrive as rows of text
rather than cells. Good enough for retrieval and for the model to read; a layout-aware extractor would be the
upgrade path.
