"""Validation rules on the models that cross stage boundaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from genai_chatbot.schemas import MAX_SUBQUESTIONS, ChatAnswer, Citation, QueryPlan


def test_query_plans_are_normalized_to_distinct_queries():
    query_plan = QueryPlan(
        decompose=True,
        queries=["  revenue   2024 ", "Revenue 2024", "", "net income", "margins", "cash", "debt"],
        response_style="table",
    )

    assert query_plan.queries == ["revenue 2024", "net income", "margins", "cash"]
    assert len(query_plan.queries) == MAX_SUBQUESTIONS
    assert query_plan.decompose


def test_a_single_query_plan_is_never_marked_decomposed():
    direct = QueryPlan(decompose=False, queries=["revenue", "net income"], response_style="prose")
    lone = QueryPlan(decompose=True, queries=["revenue", "REVENUE"], response_style="prose")

    assert (direct.decompose, direct.queries) == (False, ["revenue"])
    assert (lone.decompose, lone.queries) == (False, ["revenue"])


def test_a_plan_needs_at_least_one_query():
    with pytest.raises(ValidationError, match="no search queries"):
        QueryPlan(decompose=False, queries=["  ", ""], response_style="prose")


def test_only_answers_carry_sources_and_every_answer_has_one():
    citation = Citation(number=1, chunk_id="c1", filename="r.pdf", page_number=1, excerpt="Revenue grew.")

    with pytest.raises(ValidationError, match="at least one citation"):
        ChatAnswer(status="answered", text="Revenue grew.")
    with pytest.raises(ValidationError, match="never show sources"):
        ChatAnswer(status="unsupported", text="Not in the document.", citations=[citation])
    assert ChatAnswer(status="answered", text="Revenue grew [1].", citations=[citation]).citations == [
        citation
    ]
