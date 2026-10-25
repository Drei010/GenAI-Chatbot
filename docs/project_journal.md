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

## Feature 4 — Chroma indexing and retrieval

**Built:** `genai_chatbot/vector_store.py` (`DocumentIndex`), the `RetrievedPassage` model, and the upload
flow in `app.py` that now runs the whole ingestion graph with a progress bar, keeps the index in session
state, and drops it on reset, replacement, file removal or failure.

**Design decisions**

- *One collection per session and document.* Chroma's ephemeral clients share one in-memory store per
  process, so isolation can't come from separate clients. Each index gets a unique collection name
  (`pdfchat-<session>-<document>-<random>`) and every query also filters on `document_id`, so a passage can
  only come back from the collection it was written to, for the document it belongs to.
- *Local embeddings:* Chroma's ONNX build of `all-MiniLM-L6-v2` (384 dimensions, unit-length vectors), so
  collections use cosine distance and each passage carries a similarity score. No PyTorch, and no text
  leaves the machine for embedding. The model is downloaded once into Chroma's cache and loaded at app
  start (`st.cache_resource`) rather than during the first upload.
- *Nothing on disk:* `EphemeralClient` with telemetry off. The only file written is the model cache.
- *Up to four passages per query:* `search` caps results at four (fewer when the index is smaller) and
  returns chunk ID, filename, page, exact text, section title and score.
- *Cleanup that can't be forgotten:* `delete()` drops the collection and is idempotent, and a
  `weakref.finalize` hook drops it if a session discards the index without calling `delete()` (Streamlit
  throws away the state of closed browser sessions).
- *Upserts:* indexing the same chunks twice doesn't create duplicates, and chunks from another document are
  refused.
- *Upload errors at the UI boundary:* PDF problems keep their specific message, a document that can't be
  chunked gets "couldn't be prepared for questions", and anything unexpected gets a generic "Something went
  wrong" while the log records the exception type. The index is deleted in every failure path.
- *Testability:* `main()` takes a `router_factory`, so AppTests (and later the Playwright app) run the real
  app with a scripted model and never call a provider.

**Challenges and fixes**

- Deleted Chroma collections raise `NotFoundError` on any later call, so the index tracks its own state
  (`is_deleted`) and returns nothing instead of failing.
- `AppTest.from_function` runs a function's source as a script without `from __future__ import annotations`,
  so a type annotation in its signature raised `NameError`; the runner function is now unannotated.
- `app.py` wasn't included in coverage; added it to the coverage sources and covered the two branches it
  exposed (reruns with the same file, pages without text).

**Tests:** 205 passed (19 vector store integration tests, 11 app tests); 100% coverage of `genai_chatbot/`
and `app.py`. Manual check: uploaded a generated PDF in Chromium against the running app and confirmed the
progress box, the ready state, and log lines with stage durations and no document text.

**Gate**

- Relevant retrieval: six questions about the fictional report each rank the passage holding the answer
  first, using the real embedding model.
- Source metadata: every retrieved passage matches its indexed chunk's ID, filename, page, text and section.
- Cleanup: reset, replacement, file removal and failed uploads all leave no collection behind (checked
  against the Chroma client), and a forgotten index drops its own collection.
- Isolation: two sessions indexing the same document get separate collections, a reset in one leaves the
  other intact, and a replacement document only ever returns its own passages.

**Tradeoffs:** MiniLM is small and fast but weaker than larger embedding models on financial jargon and
numbers; retrieval quality on the Microsoft report is checked by the evaluation set in Feature 7. All
sessions share one process, so a very large number of concurrent sessions is bounded by memory.

## Feature 5 — Question planning and answer graph

**Built:** `genai_chatbot/graphs/qa.py` (the QA graph and `Conversation`), `genai_chatbot/citations.py`
(citation checking, renumbering and excerpt selection), the follow-up, planning and answer prompts in
`genai_chatbot/prompts.py`, and the plan, answer, citation and chat-turn models in `genai_chatbot/schemas.py`.

**Design decisions**

- *Graph:* `resolve_question → plan_queries → (direct) retrieve_direct | (decomposed) retrieve_decomposed →
  synthesize → validate_citations → respond`. The direct/decomposed split is a LangGraph conditional edge
  on the plan. A configuration error at any model step, or a failed synthesis, routes straight to
  `respond` with a service message.
- *Follow-ups:* the first question in a conversation skips the rewrite call. Later questions are rewritten
  into a standalone question from the latest eight messages (four exchanges); citation numbers and service
  errors are stripped from that context, and long answers are cut to 600 characters.
- *Planning:* the model returns `decompose`, up to four queries and a response style (prose, bullets or
  table). The model validator removes blank and duplicate queries, keeps at most four, and keeps
  `decompose` consistent with the query count. This lives in Pydantic rather than in JSON-schema length
  limits because Anthropic's structured outputs don't support array length constraints (the SDK strips
  them, so the model never sees them), and rejecting a five-query plan would waste a usable answer.
- *Planning failure:* one search with the question. The plan says "the original question"; this uses
  the follow-up-resolved question when there is one, since that is the user's question with its
  references filled in. If the rewrite itself fails (other than a configuration error), the raw question
  is used.
- *Retrieval:* up to four passages per query, deduplicated by chunk ID and by identical text, in query
  order. No similarity threshold: the answer model judges support, and if nothing comes back at all the
  graph abstains without calling it.
- *Citations are labels, not metadata:* passages reach the model as `<passage id="P3" section="…">` with no
  page or filename, and the model cites `[P3]` inline and in `citation_ids`. Any label that wasn't
  retrieved, or a supported answer that cites nothing, turns the reply into an abstention ("I couldn't
  match an answer to the document's text…"). Valid labels are renumbered `[1]`, `[2]` in order of first
  use, and each source's filename, page and excerpt come from the retrieved chunk.
- *Excerpts:* the sentence or table line of the passage that shares the most words and figures with the
  claims citing it (figures weigh three times as much), extended by the next line when the match is a
  heading or table row, capped at 240 characters with an ellipsis when cut. Always copied from the passage.
- *Abstention text is fixed* ("I couldn't find an answer to that in the document."), so an unsupported
  reply never shows model text that could contain an unsupported claim.
- *Conversation state:* each `Conversation` has a LangGraph `InMemorySaver` and a thread ID made of the
  session, document and a random suffix, so a re-upload of the same file after a reset starts clean. The
  saver's serializer allowlists the app's own models; LangGraph 1.x warns that unregistered types will be
  refused on load. `history()` reads the conversation back from the checkpointer, and `clear()` deletes
  the thread.
- *Unexpected errors* (say, the index failing) become "Something went wrong while answering", and an
  assistant turn is written to the thread so the history stays in question/answer pairs.
- *Logs:* one stage line each for resolve, plan, retrieve, answer and validate (route, query count,
  retrieval count, passage count, response style, citation count, outcome) and a final `answer` event, all
  under one request ID. Tests check that no question or document text appears.

**Prompt refinements**

- Passage tags first carried the section path with its ` > ` separators, which put `>` inside the tag;
  separators are now ` / `, and angle brackets and quotes in titles are replaced.
- A passage containing `</passage>` could close its tag early, so that sequence is broken up
  (`</ passage`) before formatting.
- The answer prompt says not to add page numbers, filenames or a sources list, because the app adds
  sources itself and anything the model writes there would be unchecked.
- Financial rules: copy units and periods as written, keep fiscal and calendar years apart, show and cite
  the inputs to any calculation, and report conflicting figures with their periods.

**Challenges and fixes**

- Sentence splitting broke numbered headings ("1." / "Financial Highlights") and split sentences that wrap
  across PDF lines. Units are now lines joined where the next line starts in lower case, then sentences
  split only after a word, percentage, bracket or four-digit year.
- Excerpts first extended every short match with the next sentence, which appended unrelated text to short
  complete sentences; only fragments (headings and table rows) are extended now.
- The test `Script` returned queued responder functions instead of calling them, which looked like
  malformed model output; queued functions are now called, and a script entry can also be a fixed result.

**Tests:** 258 passed (28 QA graph integration tests, 16 citation, 5 new prompt and 4 schema tests);
coverage 100% of `genai_chatbot/` and `app.py`. No test calls a provider.

**Gate:** mocked-provider integration tests cover a simple question, a decomposed question (with
deduplication and the four-query cap), follow-ups (rewrite plus the eight-message window), financial
answers (units, periods, fiscal year), summaries as cited bullets, comparison tables with a source per
row, failover on every model step, planning fallback, service errors, bad credentials without failover,
malformed output, citation validation (unknown, missing and listed-only labels) and unsupported
questions.

**Tradeoffs:** a follow-up costs three model calls (rewrite, plan, answer) and a first question two; a
single combined call would be faster but harder to test and to fall back from. Strict citation checking
can turn a correct but mis-cited answer into an abstention, which is the safer failure. Excerpts are chosen
lexically, so a paraphrased claim may get a less precise excerpt. Model calls use provider defaults (no
extended thinking), keeping the app provider-neutral and responsive.

## Feature 6 — Streamlit chat experience and browser tests

**Built:** the chat in `app.py` (question box, answers with sources, progress, abstentions and service
errors, history, reset and replacement), `tests/support/heuristic_llm.py` (a rule-based stand-in for the
model), the browser-test app `tests/e2e/fake_llm_app.py`, generated upload files
(`tests/e2e/make_fixtures.py`), `playwright.config.ts`, `tests/e2e/chat.spec.ts`, and `package.json` /
`package-lock.json` pinning `@playwright/test` 1.63.0.

**Design decisions**

- *Question box:* `st.chat_input` stays at the bottom of the page. Until a document is ready it is disabled
  and says "Upload a PDF to start asking questions". `submit_mode="disable"` locks it while a question is
  being answered, so a second question can't cut the first one off.
- *Progress:* the assistant's reply starts as a status line fed by the QA graph's progress events
  ("Reading your question in context…", "Planning the search…", "Searching the document…", "Writing the
  answer…", "Checking the sources…") and is replaced by the answer. The upload progress box is cleared the
  same way once the banner reports the outcome; before, a collapsed "Ready" box sat above the banner.
- *Answers render as Markdown* so bullet lists and tables display, but images, links and link
  definitions are removed (a PDF could plant them to make the browser fetch a URL), and `$` and `~` are
  escaped so dollar amounts don't turn into math and tildes into strikethrough. Questions and filenames are
  escaped completely and always show literally.
- *Sources:* a bordered box under each answer lists `[n] filename, page N` with the quoted excerpt, all
  taken from the retrieved passage. Abstentions show as a blue notice and service errors as a yellow
  warning, and neither shows sources.
- *History comes from the checkpointer* (`Conversation.history()`), so what the screen shows and what the
  follow-up rewrite reads are the same record. Reset and replacement delete the conversation thread along
  with the index.
- *Browser tests run the real app* with `heuristic_llm` in place of a provider. It reads the same prompts a
  model gets: it picks headings by their numbering, rewrites "it" in follow-ups from the previous question,
  plans bullets for "summarize" and a table for "compare", and answers with the passage sentence that
  shares the rarest words with the question, or abstains when nothing matches. Answers are deterministic
  and need no API key.
- *Locators use roles and names* (chat messages are groups named "Chat message from user/assistant", the
  reset button by its label, the question box by its placeholder), so the tests also check that the main
  controls are exposed to assistive technology.
- *Screenshots* of each state go to `test-results/screenshots/<browser>/`: before upload, document ready,
  answer with sources, follow-up, summary bullets, comparison table, unsupported question, invalid PDF,
  before and after reset, before and after replacement.

**Challenges and fixes**

- Playwright's `getByText` matches case-insensitive substrings, so waiting for "Sources" matched the page
  subtitle ("…page-level sources.") before any answer existed. Source checks use exact text inside the
  latest reply.
- Knowing when an answer is finished: the question box is disabled while the app works and re-enabled when
  the run ends, so `ask()` waits for the new question to appear and then for the box to come back.
- Escaping changed what AppTest reports (`northwind\.pdf`), so app tests compare text with the escapes
  removed.
- The stand-in first counted shared words equally, and the company name and year (in every passage)
  pulled a store-count passage ahead of the revenue one. Words now weigh less the more passages contain
  them, and sentences wrapped across PDF lines are joined first.

**Tests:** 268 pytest tests passed (10 new app tests: answer with sources, follow-up, bullets and tables,
unsupported, service error, reset and replacement clearing the conversation, the disabled question box,
escaping), coverage 100% of `genai_chatbot/` and `app.py`. Browser tests: 6 passed on Chromium in 21 s
(`npx playwright test --project=chromium`).

**Gate:** Playwright covers upload, a question with its source, a follow-up, summary and table formatting,
an unsupported question, an invalid PDF, reset and replacement, with screenshots for each state.
Chromium passed here. Firefox and WebKit couldn't be installed in this environment (the browser download
hosts are blocked), so they run in CI, which is set up in Feature 7.

**Tradeoffs:** the browser tests don't exercise a real model, by design; real-model quality is covered by
the evaluation set in Feature 7. Answers appear when complete rather than streaming, because citations are
checked before anything is shown. Streamlit scrolls inside its main container, so "full page" screenshots
show the viewport; each one is taken with the newest state in view.

## Feature 7 — Evaluation, documentation and CI

**Built:** the evaluation set `tests/evaluation/annual_report_questions.jsonl` (16 questions about the
Microsoft 2023 Annual Report), its harness (`harness.py`), a page locator (`locate_pages.py`), a live
runner (`run_eval.py`) and their tests; `docs/architecture.md`; the full `README.md`; the CI workflow
`.github/workflows/test.yml`; and `.python-version` (3.13).

**Design decisions**

- *Question set:* each line has the question, a category (fact, financial, comparison, summary,
  multi-part, follow-up, unsupported), whether it expects an answer or an abstention, the known facts, the
  answer points (each a list of acceptable phrasings, such as `"211.9 billion"` or `"211,915"`), evidence
  phrases from the report, and the expected source pages. Follow-ups name the question they follow and
  share its conversation. Two questions (Apple's revenue, a 2030 forecast) must be declined.
- *The set checks itself against the PDF:* a page is an expected source when it contains one of the
  question's evidence phrases, and every answer point must appear on one of those pages. `locate_pages`
  reports any phrase or point it can't find and `--write` fills in the pages, so a wrong fact or page
  shows up as a failing check instead of a silently wrong expectation. The facts were written from
  Microsoft's published fiscal 2023 results. The report couldn't be downloaded here, so its pages are
  still empty in the file and get filled by one command once the PDF is added.
- *Scoring:* an answer passes when the outcome is right (answered or declined), every point appears
  (case, spacing, typography and Markdown ignored; numbers match whole, so "7%" doesn't match "17%"), a
  cited page is an expected page, and a comparison or summary uses a table or bullets. A run passes when
  every unsupported question is declined, at least 85% of answerable questions are answered, 80% of points
  are found and 80% of answers cite an expected page; with 14 answerable questions, that allows one or two
  misses.
- *Same pipeline as the app:* the harness indexes the PDF with the ingestion graph and asks through
  `Conversation`, so the evaluation measures what users get. It deletes its index at the end.
- *Kept out of CI:* the live run is marked `live_llm` and `evaluation`, both excluded by the pytest
  configuration, and it skips without the sample or valid settings. Every other test runs with provider
  settings removed; tests marked `live_llm` keep the developer's real settings so the live run can work.
- *The harness itself is tested offline:* a run on the fictional Northwind report with the rule-based
  stand-in model covers scoring, follow-ups, reports and cleanup in every CI run.
- *CI:* one job runs ruff and pytest with a 100% coverage floor; a matrix runs Playwright on Chromium,
  Firefox and WebKit (fail-fast off) and uploads screenshots and reports. Actions use `checkout@v7`,
  `setup-node@v7` and `upload-artifact@v7`; `setup-uv` stopped publishing major tags at v8, so it is
  pinned to the v8.3.1 release commit. `.python-version` makes CI use the same Python as development.
- *Architecture doc:* module map, both graphs as diagrams, failover and failure handling, state and
  cleanup, security, logging, testing, and a table of alternatives considered.

**Challenges and fixes**

- The sample report's download hosts are blocked in this environment, so its pages couldn't be read.
  Rather than guess page numbers, the evidence and locator approach lets the file verify the set.
- The autouse fixture that removes real provider settings would have made the live test always skip; it
  now leaves tests marked `live_llm` alone.
- `astral-sh/setup-uv@v8` doesn't exist (v8 publishes only exact versions); pinned to a release commit.

**Tests:** 296 passed (28 new evaluation tests), coverage 100% of `genai_chatbot/` and `app.py`; ruff clean.
The two opt-in evaluation tests skip cleanly without the sample PDF. Playwright: 6 passed on Chromium.

**Gate:** the journal and README are complete and the CI configuration is present. Still open: the
private repository `pdf-document-chatbot-exam` (it has to be created on GitHub before the push), and
filling the expected pages from the sample PDF.

**Tradeoffs:** phrase matching can miss a correct answer worded in an unexpected way; the accepted
phrasings reduce that, and the report shows every answer for a human check. A page hit is lenient when a
figure appears on several pages (statements, notes, highlights), which is fair since any of them supports
the answer. A live run makes about 35 model calls: one outline call, then two per question and three
per follow-up.
