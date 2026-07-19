"""Unpickler hardening for the LTX prompt-embedding response."""

from __future__ import annotations

import collections
import io
import pickle

import pytest

from services.text_encoder.ltx_text_encoder import _CpuUnpickler  # noqa: SLF001


def _unpickler() -> _CpuUnpickler:
    return _CpuUnpickler(io.BytesIO(b""))


def test_find_class_rejects_arbitrary_callables() -> None:
    # The conditioning payload is a network response; find_class must not resolve
    # arbitrary importable callables (that would make unpickling an RCE vector).
    with pytest.raises(pickle.UnpicklingError):
        _unpickler().find_class("os", "system")
    with pytest.raises(pickle.UnpicklingError):
        _unpickler().find_class("builtins", "eval")


def test_find_class_allows_torch_and_stdlib_containers() -> None:
    # torch rebuild helpers + stdlib containers the real payload needs still resolve.
    assert _unpickler().find_class("collections", "OrderedDict") is collections.OrderedDict
    # The CPU-remapping storage shim is injected, not the raw torch symbol.
    assert callable(_unpickler().find_class("torch.storage", "_load_from_bytes"))
    assert callable(_unpickler().find_class("torch._utils", "_rebuild_tensor_v2"))
