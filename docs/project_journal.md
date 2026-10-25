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
