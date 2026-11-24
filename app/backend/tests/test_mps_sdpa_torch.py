"""Tests for routing Diffusers/Z-Image F.sdpa through mps-sdpa on Darwin."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch.nn.functional as F

import services.patches.mps_sdpa_torch as mps_sdpa_torch

_REAL_SDPA = F.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _restore_sdpa() -> None:
    yield
    F.scaled_dot_product_attention = _REAL_SDPA
    mps_sdpa_torch._installed = False
    mps_sdpa_torch._original_sdpa = _REAL_SDPA
    mps_sdpa_torch._sdpa_opt = None


def _mps_tensor() -> SimpleNamespace:
    return SimpleNamespace(device=SimpleNamespace(type="mps"))


def test_install_is_noop_off_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "linux")
    before = F.scaled_dot_product_attention
    mps_sdpa_torch.install()
    assert F.scaled_dot_product_attention is before
    assert mps_sdpa_torch._installed is False


def test_install_noops_when_mps_sdpa_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")
    monkeypatch.delitem(sys.modules, "mps_sdpa", raising=False)
    real_import = __import__

    def _import(name: str, *args: object, **kwargs: object):
        if name == "mps_sdpa" or name.startswith("mps_sdpa."):
            raise ImportError("test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _import)
    before = F.scaled_dot_product_attention
    mps_sdpa_torch.install()
    assert F.scaled_dot_product_attention is before
    assert mps_sdpa_torch._installed is False


def test_mps_query_routes_to_sdpa_opt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")
    seen: list[object] = []

    def fake_opt(query: object, key: object, value: object, **kwargs: object) -> str:
        seen.append((query, kwargs.get("attn_mask")))
        return "opt-out"

    fake = types.ModuleType("mps_sdpa")
    fake.sdpa_opt = fake_opt
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake)

    mps_sdpa_torch.install()
    q = _mps_tensor()
    assert F.scaled_dot_product_attention(q, q, q, attn_mask="mask") == "opt-out"
    assert seen == [(q, "mask")]


def test_cpu_query_stays_on_stock_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")
    fake = types.ModuleType("mps_sdpa")
    fake.sdpa_opt = lambda *args, **kwargs: pytest.fail("sdpa_opt must not run on CPU")
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake)

    mps_sdpa_torch.install()
    q = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    mps_sdpa_torch._original_sdpa = lambda *args, **kwargs: "stock"
    assert F.scaled_dot_product_attention(q, q, q) == "stock"


def test_stock_fallback_does_not_recurse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")
    nested: list[str] = []

    def fake_opt(query: object, key: object, value: object, **kwargs: object) -> str:
        nested.append("opt")
        return F.scaled_dot_product_attention(query, key, value)

    fake = types.ModuleType("mps_sdpa")
    fake.sdpa_opt = fake_opt
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake)

    mps_sdpa_torch.install()
    mps_sdpa_torch._original_sdpa = lambda *args, **kwargs: "from-orig"
    q = _mps_tensor()
    assert F.scaled_dot_product_attention(q, q, q) == "from-orig"
    assert nested == ["opt"]


def test_enable_gqa_bypasses_mps_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")
    fake = types.ModuleType("mps_sdpa")
    fake.sdpa_opt = lambda *args, **kwargs: pytest.fail("GQA must use stock SDPA")
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake)

    mps_sdpa_torch.install()
    mps_sdpa_torch._original_sdpa = lambda *args, **kwargs: "gqa-stock"
    q = _mps_tensor()
    assert F.scaled_dot_product_attention(q, q, q, enable_gqa=True) == "gqa-stock"


def test_sdpa_opt_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")

    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("metal failed")

    fake = types.ModuleType("mps_sdpa")
    fake.sdpa_opt = boom
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake)

    mps_sdpa_torch.install()
    mps_sdpa_torch._original_sdpa = lambda *args, **kwargs: pytest.fail("must not fall back to stock SDPA")
    q = _mps_tensor()
    with pytest.raises(RuntimeError, match="metal failed"):
        F.scaled_dot_product_attention(q, q, q)


def test_install_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_sdpa_torch.sys, "platform", "darwin")
    fake = types.ModuleType("mps_sdpa")
    fake.sdpa_opt = lambda *args, **kwargs: "opt"
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake)

    mps_sdpa_torch.install()
    first = F.scaled_dot_product_attention
    mps_sdpa_torch.install()
    assert F.scaled_dot_product_attention is first
