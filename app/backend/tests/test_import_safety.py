"""Import safety checks for test harness decoupling."""

from __future__ import annotations

from pathlib import Path


def test_conftest_does_not_stub_modules_or_import_runtime_bootstrap() -> None:
    content = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    assert "sys.modules" not in content
    assert "ltx2_server" not in content


def test_app_factory_is_importable() -> None:
    from app_factory import create_app

    assert callable(create_app)
