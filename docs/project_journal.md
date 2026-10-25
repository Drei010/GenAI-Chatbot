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

## Feature 3 — Complexity classifier and ingestion graph

**Built:** `genai_chatbot/chunking.py` (heading detection, complexity assessment, section building, recursive
and model-assisted chunking, chunk validation), `genai_chatbot/providers.py` (model construction, error
classification, the primary/fallback router), the outline prompt in `genai_chatbot/prompts.py`, the LangGraph
workflow in `genai_chatbot/graphs/ingestion.py`, and test doubles in `tests/support/fake_llm.py` plus a
fictional five-page annual report with known facts (`tests/support/documents.py`).

**Design decisions**

- *Graph shape:* `extract → classify → (simple) chunk_recursive | (complex) chunk_with_model → validate →
  index_chunks`. Conditional edges send a complex document back to `chunk_recursive` when the model call
  fails after failover, when its outline is malformed, or when its chunks fail validation. Nodes emit
  progress through LangGraph's custom stream, which the app will show in a status box.
- *Complexity rule, as planned:* more than 20 pages, three or more table-like lines (rows where figures make
  up at least 40% of the tokens), or a hierarchy of at least two heading levels across three or more strong
  headings. The assessment records which reasons applied, and the log records only the boolean.
- *Strong vs. weak headings:* numbered headings ("1.1 Revenue by Segment"), 10-K markers ("PART II",
  "Item 7") and short ALL-CAPS lines are trusted without a model. Title-case lines with no digits are only
  candidates, because in a financial report many of them are table labels.
- *Simple chunking:* sections come from strong headings and may continue across pages, but every chunk stays
  inside one page so it can be cited with one page number. Sections up to 1,000 characters stay whole;
  longer ones go through `RecursiveCharacterTextSplitter` (1,000/150, paragraph → line → sentence → word).
  Pieces under 200 characters (usually a lone heading) are folded into a neighbour when the result still
  fits, and a heading with no body joins the section that follows it.
- *Model-assisted chunking is bounded:* the model never rewrites text. It gets at most 400 candidate lines
  (ID, page, first 100 characters), picks the ones that start sections with a level from 1 to 3, and is told
  to return at most 150. The app then cuts the page text at those lines, so every chunk is still an exact
  substring of its page. An ID the model invents makes the whole outline malformed.
- *Validation before indexing:* every chunk must belong to the document, name a real page, have a unique ID,
  and match its page's text exactly. Invalid model chunks fall back to the splitter; invalid recursive chunks
  stop ingestion with an error rather than indexing something that can't be cited.
- *Failover, exactly once, and only for rate limits:* `LLMRouter` calls the primary model and, on a rate
  limit, the fallback model once. Credential, request and malformed-output errors never fail over, because
  retrying elsewhere would hide a configuration problem or double the cost of a bad prompt. SDK retries are
  off (`max_retries=0`) so a 429 reaches the router at once.
- *Provider-neutral errors:* `classify_error` walks the exception's cause chain and recognises LangChain's
  `Model*Error` types, then HTTP status codes and well-known class names. Tested against real exception
  objects from the Anthropic, OpenAI, Groq and Google SDKs.
- *Structured output:* Anthropic uses `method="json_schema"` (native structured outputs), because current
  Claude models reject the forced tool choice LangChain uses by default; other providers keep their default.
- *Bad credentials during ingestion:* the document is still indexed with the splitter, and the user gets a
  notice that names the route ("the API key for the primary model") without any provider detail, since
  questions will fail the same way until the key is fixed.

**Prompt notes**

- Candidates are listed as `id | page | text` lines rather than JSON objects, which keeps the input compact
  (an estimated 25 tokens per candidate, so about 10k tokens at the 400-candidate cap).
- v2 added the explicit ceiling of 150 sections after estimating output size: roughly 14 tokens per returned
  heading, so an unbounded answer on a long report could run past the request timeout.
- The system prompt says the candidate text is document data and any instructions in it should be ignored;
  values are inserted after templating, so braces in a PDF can't break or inject into the prompt.

**Challenges and fixes**

- The splitter emitted a lone heading as its own chunk whenever the next paragraph was too long to merge;
  `_split` now locates each piece in the source text and folds small pieces into a neighbour.
- A test fixture with templated appendix pages ("Appendix 7: Store Profile 7") had its first two lines
  stripped as running headers, because header detection ignores digits. That behaviour is right for real
  running headers, so the fixture now uses distinct text per page; the risk for real reports is noted below.
- `configure_logging` (called by the app in AppTests) left propagation off for later tests; the logger reset
  is now an autouse fixture in `tests/conftest.py`, with an `app_logs` fixture for asserting on log records.
- `fallback_reason` was missing from the logging allowlist, so the reason a document fell back was silently
  dropped from logs; added it, with a test.

**Tests:** 180 passed (111 new: 44 chunking, 47 provider and routing, 3 prompt, 17 ingestion graph
integration); coverage 100% of `genai_chatbot/`. `ruff check` and `ruff format --check` clean.

**Gate**

- Simple and complex routing: short prose stays simple and never calls a model; table-like content, section
  hierarchy and more than 20 pages are each complex, and the 20/21-page boundary is tested.
- Preserved page metadata: every chunk is checked to be an exact substring of its page, with the right
  filename, document ID and page, across a 24-page report, for both strategies.
- Malformed model output: unknown heading IDs and unparseable output both fall back to the splitter with
  reason `malformed_output`, without trying the fallback model.
- Rate-limit routing: a rate-limited primary hands the outline call to the fallback exactly once (logged
  with `fallback_used=true`); other errors never fail over.
- Deterministic fallback: when both models are rate limited, or credentials are bad, or there is no model,
  the chunks equal the splitter's output.

**Tradeoffs:** model-assisted chunking adds one model call (an estimated 10k input tokens at the candidate cap)
and some seconds to each complex upload. Header removal ignores digits, so a line that repeats on most pages with
only a number changing ("Store 7", "Store 8") is treated as a running header; real annual reports rarely
open most pages that way. Recursive overlap applies at the separator level, so paragraphs that fill a chunk
exactly don't overlap the next one.
