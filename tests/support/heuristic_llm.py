"""A rule-based stand-in for the AI provider, for browser tests and app tests.

It reads the same prompts a real model gets and answers from the passages in them by word overlap, so
the whole app can run end to end, deterministically, with no network and no API key. It is not a
model and makes no attempt to be clever: it exists to exercise the app.
"""

from __future__ import annotations

import re

from langchain_core.messages import BaseMessage

from genai_chatbot.schemas import (
    AnswerDraft,
    OutlineHeading,
    QueryPlan,
    SectionOutline,
    StandaloneQuestion,
)
from tests.support.fake_llm import FakeChatModel, Script, answer_draft, make_router, passages_in

_WORD = re.compile(r"[a-z]+|\d[\d,.]*\d|\d")
_SENTENCE = re.compile(r"(?:(?<=[a-z%)][.!?])|(?<=\d{4}[.!?]))\s+")
# fmt: off
_GENERIC = frozenset({
    "the", "and", "for", "was", "were", "are", "with", "from", "that", "this", "what", "which", "who",
    "how", "many", "much", "did", "does", "its", "about", "tell", "give", "list", "show", "year", "years",
    "fiscal", "company", "report", "document", "please",
})
# fmt: on
_FOLLOW_UP_START = re.compile(r"^(and|also|what about|how about)\b[\s,]*", re.IGNORECASE)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if (len(w) >= 3 or w[0].isdigit()) and w not in _GENERIC}


def _human(messages: list[BaseMessage]) -> str:
    return str(messages[-1].content)


def outline(messages: list[BaseMessage]) -> SectionOutline:
    """Numbered and ALL-CAPS candidates are sections, at the depth their numbering implies."""
    headings = []
    for line in _human(messages).split("\n"):
        parts = line.split(" | ", 2)
        if len(parts) != 3 or not parts[0].startswith("h"):
            continue
        candidate_id, _, text = parts
        number = re.match(r"^(\d+(?:\.\d+)*)\.?\s", text)
        if number:
            headings.append(
                OutlineHeading(candidate_id=candidate_id, level=min(number.group(1).count(".") + 1, 3))
            )
        elif text.isupper():
            headings.append(OutlineHeading(candidate_id=candidate_id, level=1))
    return SectionOutline(headings=headings)


def standalone(messages: list[BaseMessage]) -> StandaloneQuestion:
    """'And how many stores did it operate?' after a question about Northwind becomes
    'how many stores did Northwind operate?'."""
    history, latest = _human(messages).split("\n\nLatest message: ", 1)
    users = [line[len("User: ") :] for line in history.split("\n") if line.startswith("User: ")]
    question = _FOLLOW_UP_START.sub("", latest).strip()
    subject = next(iter(re.findall(r"(?<!^)\b([A-Z][a-z]+)\b", users[-1])), None) if users else None
    if subject:
        question = re.sub(r"\bits\b", f"{subject}'s", question)
        question = re.sub(r"\bit\b", subject, question)
    return StandaloneQuestion(question=question[:1].upper() + question[1:])


def plan(messages: list[BaseMessage]) -> QueryPlan:
    question = _human(messages).split("Question: ", 1)[1].strip()
    lowered = question.lower()
    if "summar" in lowered or "key facts" in lowered:
        return QueryPlan(
            decompose=True,
            queries=[
                question,
                "financial results revenue income",
                "operations stores employees",
                "sustainability",
            ],
            response_style="bullets",
        )
    style = "table" if "compare" in lowered or "table" in lowered else "prose"
    return QueryPlan(decompose=False, queries=[question], response_style=style)


def _sentences(text: str) -> list[str]:
    unwrapped = re.sub(r"\n(?=[a-z])", " ", text)  # a sentence wrapped onto the next line
    return [s.strip() for line in unwrapped.split("\n") for s in _SENTENCE.split(line) if s.strip()]


def _is_table_row(line: str) -> bool:
    return len(re.findall(r"\d[\d,.]*%?", line)) >= 2 and not line.rstrip().endswith(".")


def answer(messages: list[BaseMessage]) -> AnswerDraft:
    prompt = _human(messages)
    question = prompt.split("\nQuestion: ", 1)[1].split("\n", 1)[0]
    style = prompt.rsplit("Requested response style: ", 1)[1].strip()
    passages = passages_in(messages)
    wanted = _words(question)
    # A word found in every passage (the company's name, the year) says little about which one answers.
    spread = {word: sum(word in _words(text) for text in passages.values()) for word in wanted}

    def overlap(text: str) -> float:
        return sum(1 / spread[word] for word in wanted & _words(text))

    if style == "bullets":
        bullets = []
        for label, text in list(passages.items())[:4]:
            sentence = next((s for s in _sentences(text) if len(s.split()) >= 6), None)
            if sentence:
                bullets.append(f"- {sentence} [{label}]")
        return answer_draft("\n".join(bullets), passages, style="bullets")

    ranked = sorted(passages.items(), key=lambda item: -overlap(item[1]))
    if not ranked or overlap(ranked[0][1]) == 0:
        return answer_draft("The document doesn't cover this.", passages, status="unsupported")
    label, text = ranked[0]

    if style == "table":
        rows = [line for line in text.split("\n") if _is_table_row(line)]
        if rows:
            body = []
            for row in rows:
                name = re.split(r"\s[$(\d]", row, maxsplit=1)[0].strip()
                figures = " · ".join(f.strip() for f in re.findall(r"\$?\s?\d[\d,.]*%?", row[len(name) :]))
                body.append(f"| {name} | {figures} | [{label}] |")
            table = "| Item | Figures | Source |\n|---|---|---|\n" + "\n".join(body)
            return answer_draft(table, passages, style="table")

    best = max(_sentences(text), key=overlap)
    return answer_draft(f"{best} [{label}]", passages)


def heuristic_router():
    """A router whose primary 'model' is the rules above and that has no fallback."""
    responder = Script(
        {SectionOutline: outline, StandaloneQuestion: standalone, QueryPlan: plan, AnswerDraft: answer}
    )
    router, _, _ = make_router(responder)
    return router


__all__ = ["FakeChatModel", "answer", "heuristic_router", "outline", "plan", "standalone"]
