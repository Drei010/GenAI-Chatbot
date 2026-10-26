
# Implementation Prompt: PDF Knowledge Chatbot

## Role

Act as a senior Python application engineer specializing in Streamlit, LangChain, LangGraph, ChromaDB, RAG, prompt engineering, and testing. Follow the project’s `AGENTS.md` instructions and use the Ponytail skill for code work. Keep changes focused, but complete each feature and its tests before moving on.

## Task

Build, test, and document a Streamlit chatbot that answers natural-language questions about uploaded PDFs.

Implement the work **one feature at a time**. After each feature, run its tests, fix failures, update the project journal, and report that feature’s gate before continuing. Do not stop after producing a design or scaffold; complete the whole implementation unless a genuine external dependency or missing credential blocks progress.

## Context and required behavior

The workspace has no application files, so create the project from scratch. The application must let a user:

1. Upload a PDF. Download Microsoft’s 2023 Annual Report from its [official filing page](https://microsoft.gcs-web.com/static-files/985047d1-e200-4865-837a-c7f03ecb6212) into `samples/microsoft_2023_annual_report.pdf` as the demo document.
2. Ask questions about the document in natural language, including follow-up questions.
3. Receive concise answers grounded in document content, with sources displayed directly below each supported answer.

For every source, show the uploaded filename, the PDF’s 1-based page number, and a short excerpt copied from the retrieved text. The model may return citation IDs, but it must not invent source metadata or excerpts. Resolve each citation against retrieved passages. If the document does not support an answer, show a clear abstention and no fabricated source.

The system must handle compound or complex questions by planning retrieval and synthesis dynamically. It must preserve conversation context for follow-ups, answer financial questions with accurate units and periods, summarize key facts when asked, and format tables when a question calls for comparison.

## Architecture and interfaces

Use LangChain for provider-neutral LLM integrations and LangGraph for workflow orchestration. Build two graph workflows:

- **Document-ingestion graph:** extract and clean PDF text → classify document complexity → route to simple or complex chunking → validate chunks → embed and index in ChromaDB.
- **Question-answer graph:** use recent chat context to resolve follow-ups → plan whether to decompose the question → retrieve passages for each planned query → synthesize an answer → validate citations → return the answer and sources or abstain.

Use a LangGraph conditional route for direct, single-query questions versus decomposed questions. Cap decomposition at four subquestions and retrieve up to four passages per subquestion. Deduplicate matching chunks before synthesis. If the planning call fails after provider failover, continue with one retrieval query using the original question.

Use LangChain’s provider-neutral model factory. Configure separate primary and fallback provider identifiers, model identifiers, and API keys through Pydantic settings loaded from environment variables or Streamlit secrets. On a rate-limit response, switch to the fallback once. Do not fail over on invalid credentials or malformed requests; show a clear error. If both model calls for complex chunking fail, use the deterministic chunking fallback.

Use Pydantic models for settings and validated page, chunk, query-plan, answer, and citation data. The answer model should include answer text, response style, supported/unsupported status, and citation IDs. The query-plan model should indicate whether to decompose the question, contain at most four retrieval queries, and choose prose, bullets, or a table.

Use in-memory ChromaDB collections scoped to the active Streamlit session and document. Use local `all-MiniLM-L6-v2` embeddings and retrieve up to four passages per retrieval query. Use an in-memory LangGraph checkpointer with a unique thread ID per session/document. Use the latest eight chat messages as follow-up context. Replacing the document or resetting the chat must clear the prior thread and index. Do not persist PDFs, indexes, or chat sessions to disk.

For financial answers, preserve the period and units stated in the document, distinguish fiscal from calendar years, and include citations alongside table rows when returning a table. For summaries, use concise bullets with source references.

## Planned file structure

```text
genai-chatbot/
├── AGENTS.md
├── app.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── README.md
├── samples/
│   └── microsoft_2023_annual_report.pdf
├── docs/
│   ├── architecture.md
│   └── project_journal.md
├── genai_chatbot/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── pdf_processing.py
│   ├── chunking.py
│   ├── prompts.py
│   ├── providers.py
│   ├── vector_store.py
│   ├── observability.py
│   └── graphs/
│       ├── __init__.py
│       ├── ingestion.py
│       └── qa.py
├── tests/
│   ├── unit/
│   │   ├── test_pdf_processing.py
│   │   ├── test_chunking.py
│   │   ├── test_prompts.py
│   │   └── test_providers.py
│   ├── integration/
│   │   ├── test_ingestion_graph.py
│   │   ├── test_vector_store.py
│   │   └── test_qa_graph.py
│   ├── evaluation/
│   │   └── annual_report_questions.jsonl
│   └── e2e/
│       └── chat.spec.ts
├── playwright.config.ts
├── package.json
├── package-lock.json
└── .github/
    └── workflows/
        └── test.yml
```

If `AGENTS.md` is missing, create it at the project root with the supplied project directives: Ponytail for code work, pytest coverage, Pydantic validation, Playwright setup, and screenshots for key UI states. Use `pyproject.toml` and `uv.lock` for reproducible Python setup; use npm and the lockfile for Playwright.

## Feature sequence and acceptance gates

1. **Project foundation**

   - Create the project structure, Streamlit shell, Pydantic settings, secrets handling, README, and test configuration.
   - Add a private-safe `.gitignore` that excludes `.env`, uploaded documents, temporary Chroma data, and test artifacts.
   - **Gate:** The app starts; missing or invalid provider settings produce a helpful setup message; no secrets appear in tracked files.
2. **PDF upload and extraction**

   - Add the upload flow and extract/clean page text while retaining document filename and 1-based page numbers.
   - Detect non-PDF uploads, corrupt PDFs, empty PDFs, and scanned/textless PDFs. Explain that OCR is outside v1.
   - **Gate:** Pytest covers valid extraction and each failure case. The annual report sample can be uploaded through the app.
3. **Complexity classifier and ingestion graph**

   - Classify a document as complex if it has more than 20 pages, table-like content, or a detected section hierarchy.
   - For simple documents, keep sections up to 1,000 characters intact and recursively split longer sections with 1,000-character chunks and 150-character overlap.
   - For complex documents, run bounded, model-assisted chunking that preserves section boundaries and page metadata. If the model path fails after provider failover, use the recursive splitter.
   - **Gate:** Pytest covers simple and complex routing, preserved page metadata, malformed model output, rate-limit routing, and deterministic fallback.
4. **Chroma indexing and retrieval**

   - Embed validated chunks locally and index them in an in-memory Chroma collection scoped to the active session/document.
   - Add retrieval of up to four passages per query, retaining chunk ID, filename, page number, and text. Delete the collection on document replacement or reset.
   - **Gate:** Integration tests verify relevant retrieval, source metadata, cleanup, and isolation between sessions/documents.
5. **Question planning and answer graph**

   - Use conversation context to rewrite follow-up questions into standalone retrieval queries when needed.
   - Dynamically choose direct retrieval or decomposition into no more than four subquestions. Retrieve up to four passages per subquestion, deduplicate, then synthesize.
   - Use structured prompts for concise answers, financial values and periods, summaries, and comparison tables. Require the model to answer only from retrieved passages and return citation IDs.
   - On a rate limit, call the configured fallback provider once. If planning fails, retrieve using the original question. If synthesis has no supporting evidence, citation IDs are invalid, or both providers fail, return a clear abstention or service message.
   - Resolve valid citation IDs to the original filename, page number, and excerpt from the retrieved chunk; never use model-generated source text.
   - **Gate:** Mocked-provider integration tests cover simple queries, decomposed questions, follow-ups, financial answers, summaries, tables, failover, citation validation, and unsupported questions without live API calls.
6. **Complete the Streamlit experience**

   - Render each assistant response followed by a visible **Sources** section. Show filename, page number, and a short exact excerpt for every cited passage.
   - Show indexing progress, clear the prior chat and index on reset or replacement, and give users understandable errors for upload, provider, and processing failures.
   - **Gate:** Playwright verifies upload, follow-up question, answer and sources, summary/table formatting, unsupported question, invalid PDF, reset, and replacement. Run Chromium, Firefox, and WebKit. Capture screenshots for before/after, success, and error states.
7. **Evaluation and journal**

   - Create a small annual-report evaluation set in `tests/evaluation/annual_report_questions.jsonl` with known facts, expected answer points, and expected source pages. Keep live-LLM evaluation out of the default CI test run; document how to run it manually with configured credentials.
   - Update `docs/project_journal.md` after each feature with design decisions, challenges, fixes, prompt refinements, test results, and tradeoffs. Explain the architecture and alternatives considered in `docs/architecture.md`.
   - Add CI to run pytest and Playwright. Document installation, provider setup, app launch, sample upload, and test commands in `README.md`.
   - **Gate:** The journal and README are complete, and CI configuration is present.

## Constraints and v1 scope

- Keep one active PDF and one session-only chat; do not add multi-document libraries, persisted sessions, model parameter controls, or Docker in v1.
- Do not add OCR.
- Use local embeddings; retrieved excerpts are sent to the configured LLM provider.
- Keep automated tests deterministic with generated small PDFs and mocked LLMs. Use the Microsoft report for manual end-to-end verification.
- Log workflow stage, duration, provider route, fallback use, retrieval count, and outcome with a request/session correlation ID. Never log API keys, full PDF text, prompts containing document content, full user questions, or retrieved excerpts.
- Use accessible Streamlit controls and clear status/error messages. Do not expose implementation details in the user-facing chat.

## Output format for implementation

During work, provide brief progress updates and report each feature gate before continuing. At completion, summarize implemented features, important files, test commands/results, provider setup, run instructions, repository status, and known v1 limitations. End that report with three concise follow-up questions labeled **Q1**, **Q2**, and **Q3**.
