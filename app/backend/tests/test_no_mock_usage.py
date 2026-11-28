"""Guardrail tests for mock-free backend test suite."""

from __future__ import annotations

from pathlib import Path
import re


def test_backend_tests_do_not_use_mocking_libraries() -> None:
    tests_dir = Path(__file__).parent
    forbidden_patterns = (
        r"\bMagicMock\b",
        r"\bunittest\.mock\b",
        r"\bfrom\s+unittest\.mock\s+import\b",
        r"\bimport\s+unittest\.mock\b",
        r"(?<!\w)patch\(",
    )

    for path in tests_dir.rglob("*.py"):
        if path.name == "test_no_mock_usage.py":
            continue

        content = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert re.search(pattern, content) is None, f"Found forbidden pattern {pattern!r} in {path}"
