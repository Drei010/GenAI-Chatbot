"""Evaluate the chatbot on a question set: load the questions, run them through the app's own pipeline,
and score the answers.

Each question lists the facts a good answer states (`answer_points`, each with acceptable phrasings) and
the PDF pages that support them (`source_pages`). A question either expects an answer or an abstention.
Follow-up questions name the question they follow and share its conversation.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from genai_chatbot.documents import document_fingerprint
from genai_chatbot.graphs.ingestion import build_ingestion_graph, ingest
from genai_chatbot.graphs.qa import Conversation
from genai_chatbot.observability import Correlation, new_id
from genai_chatbot.providers import LLMRouter
from genai_chatbot.schemas import ChatAnswer, ResponseStyle
from genai_chatbot.vector_store import DocumentIndex

Category = Literal["fact", "financial", "comparison", "summary", "multi_part", "follow_up", "unsupported"]


class EvalItem(BaseModel):
    """One evaluation question and what a correct reply looks like."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str = Field(min_length=1)
    category: Category
    expect: Literal["answer", "abstain"]
    follow_up_of: str | None = None
    expected_style: ResponseStyle | None = None
    known_facts: list[str] = Field(default_factory=list)
    answer_points: list[list[str]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> EvalItem:
        if self.expect == "answer":
            if not (self.known_facts and self.answer_points and self.evidence):
                raise ValueError(f"{self.id}: an answerable question needs known facts, points and evidence")
            if any(not point or not all(v.strip() for v in point) for point in self.answer_points):
                raise ValueError(f"{self.id}: every answer point needs at least one non-blank phrasing")
        elif self.answer_points or self.evidence or self.source_pages:
            raise ValueError(f"{self.id}: a question that expects an abstention has no points or pages")
        if self.source_pages != sorted(set(self.source_pages)) or any(p < 1 for p in self.source_pages):
            raise ValueError(f"{self.id}: source pages must be distinct 1-based numbers in order")
        if (self.category == "follow_up") != (self.follow_up_of is not None):
            raise ValueError(f"{self.id}: follow-up questions, and only they, name the question they follow")
        return self


def load_items(path: Path) -> list[EvalItem]:
    """Read a JSONL question set, checking ids are unique and follow-ups come after what they follow."""
    items: list[EvalItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = EvalItem.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        known = {earlier.id for earlier in items}
        if item.id in known:
            raise ValueError(f"{path.name} line {number}: duplicate id {item.id}")
        if item.follow_up_of is not None and item.follow_up_of not in known:
            raise ValueError(f"{path.name} line {number}: {item.id} follows an unknown or later question")
        items.append(item)
    return items


def save_items(path: Path, items: Sequence[EvalItem]) -> None:
    lines = [
        json.dumps(item.model_dump(mode="json", exclude_none=True), ensure_ascii=False) for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Matching: case, spacing, typographic quotes and dashes, and Markdown emphasis don't matter, and a
# number only matches whole ("7%" doesn't match inside "17%", "9.6" doesn't match "9.68").

_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY).lower()
    return " ".join(re.sub(r"[*`]", "", text).split())


def contains(text: str, phrase: str) -> bool:
    phrase = normalize(phrase)
    pattern = re.escape(phrase)
    if phrase[:1].isalnum():
        pattern = r"(?<![\w.,])" + pattern
    if phrase[-1:].isalnum():
        pattern += r"(?![\w])"
    return re.search(pattern, normalize(text)) is not None


def point_found(text: str, point: Sequence[str]) -> bool:
    return any(contains(text, phrasing) for phrasing in point)


@dataclass(frozen=True)
class ItemResult:
    id: str
    category: str
    expect: str
    status: str
    response_style: str
    expected_style: str | None
    points_found: list[bool]
    cited_pages: list[int]
    source_pages: list[int]
    seconds: float
    answer: str

    @property
    def outcome_ok(self) -> bool:
        return self.status == ("answered" if self.expect == "answer" else "unsupported")

    @property
    def page_hit(self) -> bool | None:
        """Whether a cited page is an expected source; None when there is nothing to compare."""
        if self.status != "answered" or not self.source_pages:
            return None
        return bool(set(self.cited_pages) & set(self.source_pages))

    @property
    def style_ok(self) -> bool | None:
        if self.expected_style is None or self.status != "answered":
            return None
        return self.response_style == self.expected_style

    @property
    def passed(self) -> bool:
        return (
            self.outcome_ok
            and all(self.points_found)
            and self.page_hit is not False
            and self.style_ok is not False
        )


def score(item: EvalItem, answer: ChatAnswer, seconds: float) -> ItemResult:
    answered = answer.status == "answered"
    return ItemResult(
        id=item.id,
        category=item.category,
        expect=item.expect,
        status=answer.status,
        response_style=answer.response_style,
        expected_style=item.expected_style,
        points_found=[answered and point_found(answer.text, point) for point in item.answer_points],
        cited_pages=sorted({citation.page_number for citation in answer.citations}),
        source_pages=list(item.source_pages),
        seconds=round(seconds, 2),
        answer=answer.text,
    )


@dataclass(frozen=True)
class Targets:
    """What a run must reach to pass. Abstentions must all be right; the rest allow a miss or two."""

    answered_rate: float = 0.85
    point_recall: float = 0.8
    page_hit_rate: float = 0.8
    abstention_accuracy: float = 1.0


def _rate(values: Sequence[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


@dataclass
class EvaluationReport:
    document: str
    page_count: int
    strategy: str
    chunk_count: int
    results: list[ItemResult] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float | int | None]:
        answerable = [r for r in self.results if r.expect == "answer"]
        abstain = [r for r in self.results if r.expect == "abstain"]
        page_hits = [r.page_hit for r in self.results if r.page_hit is not None]
        styles = [r.style_ok for r in self.results if r.style_ok is not None]
        return {
            "questions": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "answered_rate": _rate([r.status == "answered" for r in answerable]),
            "point_recall": _rate([found for r in answerable for found in r.points_found]),
            "page_hit_rate": _rate(page_hits),
            "abstention_accuracy": _rate([r.outcome_ok for r in abstain]),
            "style_accuracy": _rate(styles),
            "errors": sum(r.status == "error" for r in self.results),
            "mean_seconds": round(sum(r.seconds for r in self.results) / len(self.results), 2)
            if self.results
            else None,
        }

    def shortfalls(self, targets: Targets) -> list[str]:
        """Targets this run missed, in words; empty when it passes."""
        metrics = self.metrics
        missed = []
        if metrics["errors"]:
            missed.append(f"{metrics['errors']} question(s) ended in a service error")
        for name, target in asdict(targets).items():
            value = metrics[name]
            if value is not None and value < target:
                missed.append(f"{name.replace('_', ' ')} {value:.0%} is below the {target:.0%} target")
        return missed

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "page_count": self.page_count,
            "strategy": self.strategy,
            "chunk_count": self.chunk_count,
            "metrics": self.metrics,
            "results": [
                asdict(r) | {"passed": r.passed, "page_hit": r.page_hit, "style_ok": r.style_ok}
                for r in self.results
            ],
        }

    def to_markdown(self, targets: Targets | None = None) -> str:
        m = self.metrics

        def pct(value: float | int | None) -> str:
            return "n/a" if value is None else f"{value:.0%}"

        lines = [
            f"# Evaluation: {self.document}",
            "",
            f"{self.page_count} pages, {self.chunk_count} chunks ({self.strategy} chunking). "
            f"{m['passed']} of {m['questions']} questions passed; mean {m['mean_seconds']} s per question.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answerable questions answered | {pct(m['answered_rate'])} |",
            f"| Expected answer points found | {pct(m['point_recall'])} |",
            f"| Answers citing an expected page | {pct(m['page_hit_rate'])} |",
            f"| Unsupported questions declined | {pct(m['abstention_accuracy'])} |",
            f"| Requested format used (table, bullets) | {pct(m['style_accuracy'])} |",
            f"| Service errors | {m['errors']} |",
            "",
            "| Question | Category | Result | Points | Cited pages | Expected pages | Seconds |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.results:
            result = "pass" if r.passed else f"FAIL ({r.status})"
            points = f"{sum(r.points_found)}/{len(r.points_found)}" if r.points_found else "-"
            cited = ", ".join(map(str, r.cited_pages)) or "-"
            expected = ", ".join(map(str, r.source_pages)) or "-"
            lines.append(
                f"| {r.id} | {r.category} | {result} | {points} | {cited} | {expected} | {r.seconds} |"
            )
        if targets is not None:
            missed = self.shortfalls(targets)
            lines += ["", "**Targets:** " + ("all met." if not missed else "; ".join(missed) + ".")]
        lines += ["", "## Answers", ""]
        for r in self.results:
            lines += [f"### {r.id}", "", r.answer, ""]
        return "\n".join(lines)


def run_evaluation(
    items: Sequence[EvalItem],
    filename: str,
    data: bytes,
    router: LLMRouter,
    *,
    on_result: Callable[[ItemResult], None] | None = None,
) -> EvaluationReport:
    """Index the document once, then ask every question; follow-ups continue their question's chat.

    Uses the same ingestion graph, index and question-answer graph as the app. The index is deleted at
    the end, whatever happens.
    """
    session_id = f"eval-{new_id()}"
    index = DocumentIndex(session_id, document_fingerprint(data))
    try:
        graph = build_ingestion_graph(index, router, Correlation(session_id=session_id))
        result = ingest(graph, filename, data)
        report = EvaluationReport(
            document=result.filename,
            page_count=result.page_count,
            strategy=result.strategy,
            chunk_count=result.chunk_count,
        )
        conversations: dict[str, Conversation] = {}
        for item in items:
            conversation = (
                conversations[item.follow_up_of]
                if item.follow_up_of
                else Conversation(
                    index,
                    router,
                    session_id=session_id,
                    filename=result.filename,
                    page_count=result.page_count,
                )
            )
            conversations[item.id] = conversation
            started = time.perf_counter()
            answer = conversation.ask(item.question)
            report.results.append(score(item, answer, time.perf_counter() - started))
            if on_result is not None:
                on_result(report.results[-1])
        return report
    finally:
        index.delete()
