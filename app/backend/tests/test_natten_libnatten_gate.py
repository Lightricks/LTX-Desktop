"""Flex-only natten must not count as available for cutlass-fna."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from services.patches.natten_libnatten_gate import _gate_natten_available, _has_libnatten


def test_has_libnatten_is_false_when_import_fails(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "natten", None)
    assert _has_libnatten() is False


def test_has_libnatten_is_false_for_flex_only_module(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "natten", SimpleNamespace())
    assert _has_libnatten() is False


def test_has_libnatten_is_false_when_flag_is_false(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "natten", SimpleNamespace(HAS_LIBNATTEN=False))
    assert _has_libnatten() is False


def test_has_libnatten_is_true_when_flag_set(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "natten", SimpleNamespace(HAS_LIBNATTEN=True))
    assert _has_libnatten() is True


def test_gate_clears_available_when_libnatten_missing() -> None:
    mod = SimpleNamespace(_NATTEN_AVAILABLE=True)
    _gate_natten_available(mod, has_libnatten=False)
    assert mod._NATTEN_AVAILABLE is False


def test_gate_keeps_available_when_libnatten_present() -> None:
    mod = SimpleNamespace(_NATTEN_AVAILABLE=True)
    _gate_natten_available(mod, has_libnatten=True)
    assert mod._NATTEN_AVAILABLE is True


def test_gate_leaves_missing_natten_alone() -> None:
    mod = SimpleNamespace(_NATTEN_AVAILABLE=False)
    _gate_natten_available(mod, has_libnatten=False)
    assert mod._NATTEN_AVAILABLE is False


def test_gate_clears_available_when_libnatten_missing() -> None:
    mod = SimpleNamespace(_NATTEN_AVAILABLE=True)
    _gate_natten_available(mod, has_libnatten=False)
    assert mod._NATTEN_AVAILABLE is False


def test_gate_keeps_available_when_libnatten_present() -> None:
    mod = SimpleNamespace(_NATTEN_AVAILABLE=True)
    _gate_natten_available(mod, has_libnatten=True)
    assert mod._NATTEN_AVAILABLE is True


def test_gate_leaves_missing_natten_alone() -> None:
    mod = SimpleNamespace(_NATTEN_AVAILABLE=False)
    _gate_natten_available(mod, has_libnatten=False)
    assert mod._NATTEN_AVAILABLE is False
