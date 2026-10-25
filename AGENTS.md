# AGENTS.md

Project directives for anyone, human or AI agent, who changes this repository.

## Code work

- Use the **Ponytail** skill for code work when your environment provides it. Where it isn't available, keep
  to its intent: read the surrounding code first, make small focused changes, and finish a feature and its
  tests before starting the next one.
- Work one feature at a time. When a feature lands, add an entry to `docs/project_journal.md` covering design
  decisions, challenges, fixes, prompt changes, test results and tradeoffs.
- Keep `README.md` and `docs/architecture.md` in step with behavior changes.

## Tests and coverage

- Every module in `genai_chatbot/` needs pytest coverage. Run `uv run pytest --cov=genai_chatbot` before
  committing and don't let coverage drop.
- Automated tests must be deterministic: generate small PDFs inside the tests and use the fake LLM in
  `tests/support/fake_llm.py`. The default test run never calls a real provider; live checks carry the
  `live_llm` marker and run only on request.

## Evaluation

- `tests/evaluation/annual_report_questions.jsonl` is checked against the sample report with
  `uv run python -m tests.evaluation.locate_pages`; rerun it with `--write` after changing a question.
- Live evaluation (`uv run python -m tests.evaluation.run_eval`) calls real providers and is never part of
  CI.

## Validation

- Validate data that crosses a boundary with Pydantic models (`genai_chatbot/config.py`,
  `genai_chatbot/schemas.py`): settings, pages, chunks, query plans, answers and citations.
- Treat model output as untrusted until it validates. Citations are resolved against retrieved passages;
  never display source text the model wrote.

## Playwright

- Browser tests live in `tests/e2e/` and run against `tests/e2e/fake_llm_app.py`, which swaps in the
  deterministic, rule-based stand-in model from `tests/support/heuristic_llm.py`.
- Setup: `npm ci` then `npx playwright install --with-deps chromium firefox webkit`.
- Run `npx playwright test` to cover Chromium, Firefox and WebKit.

## Screenshots

- Capture screenshots of key UI states in the e2e tests: empty state, indexed document, answer with sources,
  summary and table answers, abstention, error states, reset and replacement. They are written to
  `test-results/screenshots/<browser>/` and uploaded as a CI artifact.

## Privacy and secrets

- Never commit secrets, `.env`, `.streamlit/secrets.toml`, or uploaded documents.
- Never log API keys, PDF text, prompts that contain document content, full user questions, or excerpts.
  Use `genai_chatbot.observability`, which only accepts an allowlist of scalar fields.
