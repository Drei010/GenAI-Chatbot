# PDF Knowledge Chatbot

A Streamlit chatbot that answers natural-language questions about an uploaded PDF, with sources shown below
every answer. Work in progress: see `docs/project_journal.md` for status.

## Quick start

```bash
uv sync
cp .env.example .env   # then fill in your provider keys
uv run streamlit run app.py
```

## Tests

```bash
uv run pytest                      # unit and integration tests, no provider calls
npm ci                             # browser tests: install Playwright once
npx playwright install --with-deps chromium firefox webkit
npx playwright test                # Chromium, Firefox and WebKit, with a stand-in model
```
