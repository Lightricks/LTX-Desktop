"""Unpickler hardening for the LTX prompt-embedding response."""

from __future__ import annotations

import collections
import io
import pickle

import pytest
import torch

from services.text_encoder.ltx_text_encoder import (  # noqa: SLF001
    _ALLOWED_PICKLE_GLOBALS,
    _CpuUnpickler,
    _first_embedding_tensor,
)


def _unpickler() -> _CpuUnpickler:
    return _CpuUnpickler(io.BytesIO(b""))


def test_find_class_rejects_arbitrary_callables() -> None:
    # The conditioning payload is a network response; find_class must not resolve
    # arbitrary importable callables (that would make unpickling an RCE vector).
    with pytest.raises(pickle.UnpicklingError, match="disallowed pickle global"):
        _unpickler().find_class("os", "system")
    with pytest.raises(pickle.UnpicklingError, match="disallowed pickle global"):
        _unpickler().find_class("builtins", "eval")


def test_find_class_rejects_other_torch_symbols() -> None:
    # A prefix allowlist of torch.* would still admit jit/package gadgets.
    with pytest.raises(pickle.UnpicklingError, match="disallowed pickle global"):
        _unpickler().find_class("torch.jit", "script")
    with pytest.raises(pickle.UnpicklingError, match="disallowed pickle global"):
        _unpickler().find_class("torch", "Tensor")


def test_find_class_allows_only_tensor_rebuild_symbols() -> None:
    assert _ALLOWED_PICKLE_GLOBALS == frozenset(
        {
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch.storage", "_load_from_bytes"),
            ("collections", "OrderedDict"),
            ("_codecs", "encode"),
        }
    )
    assert _unpickler().find_class("collections", "OrderedDict") is collections.OrderedDict
    assert _unpickler().find_class("_codecs", "encode") is __import__("_codecs").encode
    # The CPU-remapping storage shim is injected, not the raw torch symbol.
    assert callable(_unpickler().find_class("torch.storage", "_load_from_bytes"))
    assert callable(_unpickler().find_class("torch._utils", "_rebuild_tensor_v2"))


def test_unpickler_roundtrips_nested_tensor_payload() -> None:
    embeddings = torch.randn(1, 8, 4096 + 384, dtype=torch.bfloat16)
    payload = pickle.dumps([[embeddings]])
    loaded = _CpuUnpickler(io.BytesIO(payload)).load()
    recovered = _first_embedding_tensor(loaded)
    assert recovered.shape == embeddings.shape
    assert recovered.dtype == embeddings.dtype
    assert torch.equal(recovered.float(), embeddings.float())


def test_first_embedding_tensor_rejects_non_tensor_nests() -> None:
    with pytest.raises(pickle.UnpicklingError, match="container"):
        _first_embedding_tensor("nope")
    with pytest.raises(pickle.UnpicklingError, match="row"):
        _first_embedding_tensor([torch.zeros(1)])
    with pytest.raises(pickle.UnpicklingError, match="not a tensor"):
        _first_embedding_tensor([["nope"]])
