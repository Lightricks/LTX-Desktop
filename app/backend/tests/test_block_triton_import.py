"""block_triton_import hides Triton so ltx-core VAE kernels use their non-Triton path."""

from __future__ import annotations

import importlib
import sys

import pytest

from inference.triton_block import block_triton_import


def test_block_triton_import_makes_import_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.delitem(sys.modules, "triton.language", raising=False)

    assert block_triton_import() is True

    with pytest.raises(ImportError):
        importlib.import_module("triton")
    with pytest.raises(ImportError):
        importlib.import_module("triton.language")


def test_block_triton_import_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "triton", None)  # pyright: ignore[reportArgumentType]

    assert block_triton_import() is True


def test_block_triton_import_reports_when_triton_already_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = type(sys)("triton")
    monkeypatch.setitem(sys.modules, "triton", fake)

    assert block_triton_import() is False
    assert sys.modules["triton"] is fake
