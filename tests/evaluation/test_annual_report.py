"""The annual-report evaluation set. Its shape is checked in every run; checking it against the sample
report and running it against live models happen only on request (see README, "Evaluation")."""

from __future__ import annotations

from typing import get_args

import pytest

from genai_chatbot.config import load_settings
from genai_chatbot.extraction import extract_document
from tests.evaluation.harness import Category, load_items
from tests.evaluation.locate_pages import QUESTIONS, locate, problems
from tests.evaluation.run_eval import TARGETS, evaluate_sample, write_report
from tests.support.paths import SAMPLE_REPORT

NEEDS_SAMPLE = pytest.mark.skipif(
    not SAMPLE_REPORT.exists(), reason="add samples/microsoft_2023_annual_report.pdf (see README)"
)


def test_the_question_set_covers_every_kind_of_question() -> None:
    items = load_items(QUESTIONS)

    assert {item.category for item in items} == set(get_args(Category))
    assert sum(item.expect == "abstain" for item in items) >= 2
    assert len(items) <= 20, "keep it small: every live run costs tokens"


@pytest.mark.evaluation
@NEEDS_SAMPLE
def test_facts_and_pages_match_the_sample_report() -> None:
    items = load_items(QUESTIONS)
    document = extract_document(SAMPLE_REPORT.read_bytes(), SAMPLE_REPORT.name)

    located = locate(items, document.pages)

    assert problems(located) == []
    saved = {item.id: item.source_pages for item in items if item.expect == "answer"}
    assert saved == {entry.item_id: entry.pages for entry in located}, (
        "source pages are out of date: run `uv run python -m tests.evaluation.locate_pages --write`"
    )


@pytest.mark.live_llm
@pytest.mark.evaluation
@NEEDS_SAMPLE
def test_live_models_meet_the_evaluation_targets() -> None:
    loaded = load_settings()
    if loaded.settings is None:
        pytest.skip(
            "provider settings are missing or invalid: " + ", ".join(i.setting for i in loaded.issues)
        )

    report = evaluate_sample(loaded.settings)
    markdown, _ = write_report(report)

    assert report.shortfalls(TARGETS) == [], f"see {markdown}"
