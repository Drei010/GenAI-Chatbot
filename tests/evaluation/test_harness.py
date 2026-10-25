"""The evaluation harness, run offline on the fictional report with the rule-based stand-in model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genai_chatbot.schemas import ChatAnswer, Citation
from genai_chatbot.vector_store import get_client
from tests.evaluation import locate_pages, run_eval
from tests.evaluation.harness import (
    EvalItem,
    EvaluationReport,
    Targets,
    contains,
    load_items,
    run_evaluation,
    save_items,
    score,
)
from tests.support.heuristic_llm import heuristic_router

EMPLOYEES = EvalItem(
    id="employees",
    question="How many people did Northwind employ?",
    category="fact",
    expect="answer",
    known_facts=["Northwind employed 2,340 people as of June 30, 2024."],
    answer_points=[["2,340"]],
    evidence=["2,340 people"],
    source_pages=[3],
)
NORTHWIND_ITEMS = [
    EMPLOYEES,
    EvalItem(
        id="stores",
        question="And how many retail stores did it operate?",
        category="follow_up",
        follow_up_of="employees",
        expect="answer",
        known_facts=["Northwind operated 186 retail stores at the end of fiscal year 2024."],
        answer_points=[["186"]],
        evidence=["186 retail stores"],
        source_pages=[3],
    ),
    EvalItem(
        id="segments",
        question="Compare revenue by segment in a table",
        category="comparison",
        expect="answer",
        expected_style="table",
        known_facts=["Segment revenue: retail stores $251.4M, subscriptions $96.7M, wholesale $64.5M."],
        answer_points=[["251.4"], ["96.7"], ["64.5"]],
        evidence=["Retail stores $ 251.4"],
        source_pages=[2],
    ),
    EvalItem(
        id="capital", question="What is the capital of France?", category="unsupported", expect="abstain"
    ),
]


def answered(text: str, *pages: int, style: str = "prose") -> ChatAnswer:
    citations = [
        Citation(number=n, chunk_id=f"c{n}", filename="r.pdf", page_number=page, excerpt="…")
        for n, page in enumerate(pages, start=1)
    ]
    return ChatAnswer(status="answered", response_style=style, text=text, citations=citations)


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("Revenue grew 7%.", "7%", True),
        ("Revenue grew 17%.", "7%", False),
        ("Diluted EPS was $9.68.", "9.68", True),
        ("Diluted EPS was $9.68.", "9.6", False),
        ("about 1,221,000 people", "221,000", False),
        ("approximately **221,000** people", "221,000", True),
        ("Revenue  was $211.9\nbillion", "$211.9 billion", True),
        ("Fiscal Year—2023", "fiscal year-2023", True),
        ("Microsoft’s cloud", "microsoft's cloud", True),
    ],
)
def test_phrases_match_whole_numbers_ignoring_case_spacing_and_typography(text, phrase, expected) -> None:
    assert contains(text, phrase) is expected


def test_a_pass_needs_the_outcome_every_point_and_an_expected_page() -> None:
    assert score(EMPLOYEES, answered("It employed 2,340 people [1].", 3), 1.0).passed
    assert not score(EMPLOYEES, answered("It employed 2,340 people [1].", 5), 1.0).passed
    assert not score(EMPLOYEES, answered("It employed many people [1].", 3), 1.0).passed

    declined = score(EMPLOYEES, ChatAnswer(status="unsupported", text="No."), 1.0)
    assert not declined.passed and declined.points_found == [False] and declined.page_hit is None

    capital = NORTHWIND_ITEMS[-1]
    assert score(capital, ChatAnswer(status="unsupported", text="No."), 1.0).passed
    assert not score(capital, answered("Paris [1].", 1), 1.0).passed


def test_the_requested_format_is_part_of_a_pass() -> None:
    segments = NORTHWIND_ITEMS[2]
    text = "| Segment | Revenue | Source |\n|---|---|---|\n| Retail | 251.4, 96.7, 64.5 | [1] |"

    assert score(segments, answered(text, 2, style="table"), 1.0).passed
    assert not score(segments, answered(text, 2, style="prose"), 1.0).passed


def test_an_evaluation_runs_through_the_app_pipeline_and_cleans_up(northwind_pdf: bytes) -> None:
    seen = []

    report = run_evaluation(
        NORTHWIND_ITEMS, "northwind.pdf", northwind_pdf, heuristic_router(), on_result=seen.append
    )

    assert [result.id for result in seen] == ["employees", "stores", "segments", "capital"]
    metrics = report.metrics
    assert metrics["passed"] == 4 and metrics["errors"] == 0
    assert {
        k: metrics[k] for k in ("answered_rate", "point_recall", "page_hit_rate", "abstention_accuracy")
    } == {
        "answered_rate": 1.0,
        "point_recall": 1.0,
        "page_hit_rate": 1.0,
        "abstention_accuracy": 1.0,
    }
    assert report.shortfalls(Targets()) == []
    assert (report.page_count, report.strategy) == (5, "model_assisted")
    assert not [c for c in get_client().list_collections() if c.name.startswith("pdfchat-eval-")]


def test_reports_show_every_question_and_the_targets_missed() -> None:
    report = EvaluationReport(document="r.pdf", page_count=3, strategy="recursive", chunk_count=4)
    report.results += [
        score(EMPLOYEES, answered("2,340 people [1].", 3), 2.0),
        score(EMPLOYEES.model_copy(update={"id": "again"}), ChatAnswer(status="error", text="Busy."), 4.0),
    ]

    missed = report.shortfalls(Targets())
    markdown = report.to_markdown(Targets())

    assert missed == [
        "1 question(s) ended in a service error",
        "answered rate 50% is below the 85% target",
        "point recall 50% is below the 80% target",
    ]
    assert "| employees | fact | pass | 1/1 | 3 | 3 | 2.0 |" in markdown
    assert "| again | fact | FAIL (error) | 0/1 | - | 3 | 4.0 |" in markdown
    assert "**Targets:** 1 question(s) ended in a service error;" in markdown
    data = report.to_dict()
    assert data["metrics"]["mean_seconds"] == 3.0
    assert [r["passed"] for r in data["results"]] == [True, False]


def _write(path: Path, *lines: dict) -> Path:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([EMPLOYEES.model_dump(), EMPLOYEES.model_dump()], "line 2: duplicate id employees"),
        ([NORTHWIND_ITEMS[1].model_dump()], "line 1: stores follows an unknown or later question"),
        ([EMPLOYEES.model_dump() | {"evidence": []}], "needs known facts, points and evidence"),
        ([EMPLOYEES.model_dump() | {"answer_points": [[" "]]}], "at least one non-blank phrasing"),
        ([EMPLOYEES.model_dump() | {"expect": "abstain"}], "has no points or pages"),
        ([EMPLOYEES.model_dump() | {"source_pages": [3, 2]}], "distinct 1-based numbers in order"),
        ([EMPLOYEES.model_dump() | {"category": "follow_up"}], "name the question they follow"),
    ],
)
def test_question_sets_are_validated_when_loaded(tmp_path: Path, lines, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_items(_write(tmp_path / "questions.jsonl", *lines))


def test_question_sets_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"

    save_items(path, NORTHWIND_ITEMS)

    assert load_items(path) == NORTHWIND_ITEMS
    assert "follow_up_of" not in path.read_text().splitlines()[0], "unset fields are left out"


def test_pages_are_located_and_facts_checked_against_the_document(tmp_path, northwind_pdf) -> None:
    pdf = tmp_path / "northwind.pdf"
    pdf.write_bytes(northwind_pdf)
    unplaced = [item.model_copy(update={"source_pages": []}) for item in NORTHWIND_ITEMS]
    questions = tmp_path / "questions.jsonl"
    save_items(questions, unplaced)

    assert locate_pages.main(["--pdf", str(pdf), "--questions", str(questions), "--write"]) == 0

    assert [item.source_pages for item in load_items(questions)] == [[3], [3], [2], []]


def test_wrong_evidence_and_unsupported_points_are_reported(northwind) -> None:
    item = EMPLOYEES.model_copy(
        update={"evidence": ["2,340 people", "9,999 people"], "answer_points": [["2,341"]]}
    )

    (located,) = locate_pages.locate([item], northwind.pages)

    assert located.pages == [3]
    assert locate_pages.problems([located]) == [
        "employees: evidence not found: '9,999 people'",
        "employees: answer point not on its pages: 2,341",
    ]


def test_locating_pages_needs_the_document(tmp_path: Path, capsys) -> None:
    assert locate_pages.main(["--pdf", str(tmp_path / "missing.pdf")]) == 2
    assert "Add the sample report first" in capsys.readouterr().out


def test_the_live_run_explains_what_is_missing(tmp_path: Path, capsys, valid_settings_env) -> None:
    assert run_eval.main(["--pdf", str(tmp_path / "missing.pdf")]) == 2
    assert "Add the sample report first" in capsys.readouterr().out


def test_the_live_run_explains_missing_settings(capsys) -> None:
    assert run_eval.main([]) == 2
    assert "PRIMARY_LLM_API_KEY is missing." in capsys.readouterr().out


def test_the_live_run_writes_its_report(
    tmp_path, capsys, monkeypatch, valid_settings_env, northwind_pdf
) -> None:
    monkeypatch.setattr(run_eval, "build_router", lambda settings: heuristic_router())
    pdf = tmp_path / "northwind.pdf"
    pdf.write_bytes(northwind_pdf)
    questions = tmp_path / "questions.jsonl"
    save_items(questions, NORTHWIND_ITEMS)

    code = run_eval.main(["--pdf", str(pdf), "--questions", str(questions), "--out", str(tmp_path / "out")])

    output = capsys.readouterr().out
    assert code == 0, output
    assert "pass  stores (answered, 1/1 points, cited pages 3)" in output
    assert "4 of 4 passed." in output
    report = json.loads((tmp_path / "out" / "annual_report.json").read_text())
    assert report["metrics"]["passed"] == 4
    assert (tmp_path / "out" / "annual_report.md").read_text().startswith("# Evaluation: northwind.pdf")
