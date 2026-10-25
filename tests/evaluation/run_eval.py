"""Run the evaluation set against the sample report with your configured models.

    uv run python -m tests.evaluation.run_eval

Reads provider settings the way the app does (environment variables, `.env` or Streamlit secrets),
indexes samples/microsoft_2023_annual_report.pdf with the app's own pipeline, asks every question in
tests/evaluation/annual_report_questions.jsonl, prints one line per question and writes the full report
to test-results/evaluation/. It calls the providers for real, so it costs tokens; the default test run
never does. Exits with 1 when a target is missed and 2 when it can't start.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from genai_chatbot.config import AppSettings, load_settings
from genai_chatbot.providers import build_router
from tests.evaluation.harness import EvaluationReport, ItemResult, Targets, load_items, run_evaluation
from tests.evaluation.locate_pages import QUESTIONS
from tests.support.paths import PROJECT_ROOT, SAMPLE_REPORT

REPORT_DIR = PROJECT_ROOT / "test-results" / "evaluation"
TARGETS = Targets()


def evaluate_sample(
    settings: AppSettings,
    *,
    pdf: Path = SAMPLE_REPORT,
    questions: Path = QUESTIONS,
    on_result: Callable[[ItemResult], None] | None = None,
) -> EvaluationReport:
    items = load_items(questions)
    return run_evaluation(items, pdf.name, pdf.read_bytes(), build_router(settings), on_result=on_result)


def write_report(report: EvaluationReport, directory: Path = REPORT_DIR) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / "annual_report.md"
    data = directory / "annual_report.json"
    markdown.write_text(report.to_markdown(TARGETS) + "\n", encoding="utf-8")
    data.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return markdown, data


def _print_result(result: ItemResult) -> None:
    points = f"{sum(result.points_found)}/{len(result.points_found)} points" if result.points_found else ""
    pages = f"cited pages {', '.join(map(str, result.cited_pages))}" if result.cited_pages else ""
    details = ", ".join(part for part in (result.status, points, pages) if part)
    print(f"{'pass' if result.passed else 'FAIL'}  {result.id} ({details}) {result.seconds}s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdf", type=Path, default=SAMPLE_REPORT, help="the sample report")
    parser.add_argument("--questions", type=Path, default=QUESTIONS, help="the JSONL question set")
    parser.add_argument("--out", type=Path, default=REPORT_DIR, help="where to write the report")
    args = parser.parse_args(argv)

    loaded = load_settings()
    if loaded.settings is None:
        print("Provider settings need attention (see README, 'Provider setup'):")
        for issue in loaded.issues:
            print(f"  {issue.setting} {issue.problem}")
        return 2
    if not args.pdf.exists():
        print(f"{args.pdf} doesn't exist. Add the sample report first (see README, 'Sample document').")
        return 2

    report = evaluate_sample(loaded.settings, pdf=args.pdf, questions=args.questions, on_result=_print_result)
    markdown, _ = write_report(report, args.out)
    missed = report.shortfalls(TARGETS)
    print(f"{report.metrics['passed']} of {report.metrics['questions']} passed. Report: {markdown}")
    for shortfall in missed:
        print(f"Missed target: {shortfall}")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
