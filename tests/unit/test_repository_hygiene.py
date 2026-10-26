"""Guards for the private-repository requirements: no secrets or private documents get tracked."""

from __future__ import annotations

import re
import subprocess

import pytest

from tests.support.paths import PROJECT_ROOT

KEY_PATTERNS = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Groq key": re.compile(r"gsk_[A-Za-z0-9]{40,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
}


def git(*args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=False)
    except OSError:
        pytest.skip("git is not available")


@pytest.fixture(scope="module")
def require_git_checkout() -> None:
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout")


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".streamlit/secrets.toml",
        "uploads/contract.pdf",
        "notes/private-report.pdf",
        "notes/2023_Annual_Report.docx",
        "notes/old-minutes.doc",
        "chroma/chroma.sqlite3",
        "test-results/screenshots/chromium/answer.png",
        "playwright-report/index.html",
        ".coverage",
    ],
)
def test_private_and_generated_paths_are_ignored(require_git_checkout: None, path: str) -> None:
    assert git("check-ignore", "-q", "--no-index", path).returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", [".env.example", "samples/microsoft_2023_annual_report.pdf"])
def test_shareable_files_are_not_ignored(require_git_checkout: None, path: str) -> None:
    assert git("check-ignore", "-q", "--no-index", path).returncode == 1, f"{path} should be trackable"


def test_tracked_files_contain_no_api_keys(require_git_checkout: None) -> None:
    listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout.decode()
    paths = [PROJECT_ROOT / name for name in listed.split("\0") if name]
    findings = []
    for path in paths:
        if not path.is_file() or path.suffix in {".pdf", ".png", ".jpg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        name = path.relative_to(PROJECT_ROOT)
        findings += [f"{name}: {label}" for label, rx in KEY_PATTERNS.items() if rx.search(text)]
    assert findings == []
