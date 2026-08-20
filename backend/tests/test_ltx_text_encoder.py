"""Unpickler hardening and `/v1/prompt-embedding` request shape."""

from __future__ import annotations

import collections
import io
import json
import pickle
import struct
from pathlib import Path

import pytest
import torch

from runtime_config.ltx_api_text_encoder_ids import LTX_2_5_API_PROMPT_EMBEDDING_MODEL
from runtime_config.model_download_specs import get_ltx_model_spec
from services.text_encoder.ltx_text_encoder import (  # noqa: SLF001
    LTXTextEncoder,
    _ALLOWED_PICKLE_GLOBALS,
    _CpuUnpickler,
    _first_embedding_tensor,
)
from tests.fakes.services import FakeHTTPClient, FakeResponse


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


def _embedding_response_bytes() -> bytes:
    embeddings = torch.randn(1, 8, 4096 + 384, dtype=torch.bfloat16)
    return pickle.dumps([[embeddings]])


def _write_safetensors_with_metadata(path: Path, metadata: dict[str, str]) -> None:
    header = {
        "__metadata__": metadata,
        "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00\x00\x00")


def _encoder_with_queued_embedding(http: FakeHTTPClient) -> LTXTextEncoder:
    http.queue("post", FakeResponse(status_code=200, content=_embedding_response_bytes()))
    return LTXTextEncoder(
        device=torch.device("cpu"),
        http=http,
        ltx_api_base_url="https://api.ltx.video",
    )


def test_ltx_2_5_spec_uses_prompt_embedding_model_not_fernet_model_id() -> None:
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    assert spec.api_prompt_embedding_model == LTX_2_5_API_PROMPT_EMBEDDING_MODEL
    assert spec.api_prompt_embedding_model == {
        "ltx_version": "2.5.0",
        "gemma_version": "gemma4-12b-ltx-v1",
    }
    assert get_ltx_model_spec("ltx-2.3-22b-distilled-1.1").api_prompt_embedding_model is None


def test_encode_via_api_sends_model_selector_without_model_id(tmp_path: Path) -> None:
    http = FakeHTTPClient()
    encoder = _encoder_with_queued_embedding(http)
    checkpoint = tmp_path / "ltx-2.5.safetensors"
    checkpoint.write_bytes(b"not-a-checkpoint")

    result = encoder.encode_via_api(
        prompt="A beautiful sunset",
        api_key="key",
        checkpoint_path=str(checkpoint),
        enhance_prompt=True,
        api_model=LTX_2_5_API_PROMPT_EMBEDDING_MODEL,
    )

    assert result is not None
    assert http.calls[0].url == "https://api.ltx.video/v1/prompt-embedding"
    assert http.calls[0].json_payload == {
        "prompt": "A beautiful sunset",
        "enhance_prompt": True,
        "model": {
            "ltx_version": "2.5.0",
            "gemma_version": "gemma4-12b-ltx-v1",
        },
    }
    assert "model_id" not in http.calls[0].json_payload


def test_encode_via_api_sends_checkpoint_model_id_for_legacy_2_3(tmp_path: Path) -> None:
    http = FakeHTTPClient()
    encoder = _encoder_with_queued_embedding(http)
    checkpoint = tmp_path / "ltx-2.3.safetensors"
    _write_safetensors_with_metadata(checkpoint, {"encrypted_wandb_properties": "legacy-model-id"})

    result = encoder.encode_via_api(
        prompt="A beautiful sunset",
        api_key="key",
        checkpoint_path=str(checkpoint),
        enhance_prompt=False,
    )

    assert result is not None
    assert http.calls[0].json_payload == {
        "prompt": "A beautiful sunset",
        "enhance_prompt": False,
        "model_id": "legacy-model-id",
    }
    assert "model" not in http.calls[0].json_payload


def test_encode_via_api_model_selector_wins_over_checkpoint_model_id(tmp_path: Path) -> None:
    http = FakeHTTPClient()
    encoder = _encoder_with_queued_embedding(http)
    checkpoint = tmp_path / "ltx-2.5.safetensors"
    _write_safetensors_with_metadata(checkpoint, {"encrypted_wandb_properties": "retired-fernet-blob"})

    encoder.encode_via_api(
        prompt="prompt",
        api_key="key",
        checkpoint_path=str(checkpoint),
        enhance_prompt=False,
        api_model=LTX_2_5_API_PROMPT_EMBEDDING_MODEL,
    )

    payload = http.calls[0].json_payload
    assert payload is not None
    assert "model" in payload
    assert "model_id" not in payload


def test_encode_via_api_skips_request_when_no_selector(tmp_path: Path) -> None:
    http = FakeHTTPClient()
    encoder = LTXTextEncoder(
        device=torch.device("cpu"),
        http=http,
        ltx_api_base_url="https://api.ltx.video",
    )
    checkpoint = tmp_path / "empty.safetensors"
    checkpoint.write_bytes(b"not-a-checkpoint")

    result = encoder.encode_via_api(
        prompt="prompt",
        api_key="key",
        checkpoint_path=str(checkpoint),
        enhance_prompt=False,
    )

    assert result is None
    assert http.calls == []
