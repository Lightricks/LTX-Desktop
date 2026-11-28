"""Weight-free tests for the Mac worker's host side. Metal execution is tested on device."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

MODULE = Path(__file__).resolve().parents[1] / "services/h3_metal/__init__.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("test_spacespeak_metal", MODULE)
    module = importlib.util.module_from_spec(spec)
    # Import outside the worker without registering any real Comfy node.
    with patch("sys.platform", "linux"):
        spec.loader.exec_module(module)
    return module


def test_feature_gate_skips_other_generations_and_honors_override(worker, monkeypatch):
    monkeypatch.setattr(worker.sys, "platform", "darwin")
    monkeypatch.setattr(worker.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(worker.platform, "mac_ver", lambda: ("26.5", (), ""))
    for brand in ("Apple M5", "Apple M5 Pro", "Apple M5 Max"):
        monkeypatch.setattr(worker.subprocess, "check_output", lambda *a, **k: brand)
        assert worker.hardware_reason() is None
    monkeypatch.setattr(worker.subprocess, "check_output", lambda *a, **k: "Apple M4 Max")
    assert "M4 Max" in worker.hardware_reason()
    monkeypatch.setenv("LTX_H3_M5_INT8", "0")
    assert "disabled" in worker.hardware_reason()


@pytest.mark.parametrize("channel_scales", [False, True])
def test_real_comfy_quantizer_chunked_dispatch_and_output_assembly(worker, channel_scales):
    quantization = pytest.importorskip("comfy_kitchen.backends.eager.quantization")
    utils = pytest.importorskip("comfy_kitchen.tensor.int8_utils")
    gen = torch.Generator().manual_seed(10)
    # Rank-three non-contiguous input; output slices after the first 1024 rows
    # share one output allocation and must respect their storage offsets.
    x = torch.randn((67, 17, 64), generator=gen).transpose(0, 1)
    weight = torch.randint(-127, 128, (256, 64), dtype=torch.int8, generator=gen)
    scale = torch.rand(256 if channel_scales else 1, generator=gen)
    bias = torch.randn(256, generator=gen) if channel_scales else None
    runtime = worker.NativeInt8.__new__(worker.NativeInt8)
    runtime._hadamard = utils._build_hadamard
    runtime._quantize = quantization.quantize_and_rotate_rowwise
    submitted = []

    def shader(a, w, out, rows, cols, bias_buf, m, n, k, scales, has_bias, *, threads, group_size, arg_casts):
        submitted.append((m, out.storage_offset()))
        assert (n, k, scales, has_bias) == (256, 64, scale.numel(), int(channel_scales))
        assert threads == (256, (m + 31) // 32) and group_size == (128, 1)
        assert arg_casts == "int32"
        result = (a.int() @ w.int().T).float() * (rows * cols)
        out.copy_(result + bias_buf if has_bias else result)

    runtime._kernels = {torch.float32: shader}
    actual = runtime.linear(x, weight, scale, bias, 64)
    expected = quantization.int8_linear(x, weight, scale, bias=bias, out_dtype=x.dtype,
                                        convrot=True, convrot_groupsize=64)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert submitted == [(1024, 0), (115, 1024 * 256)]


def test_dispatched_marker_only_once_per_executing_prompt(worker, caplog):
    runtime = worker.NativeInt8.__new__(worker.NativeInt8)
    runtime.native_calls = 0
    runtime._last_prompt = None
    with patch.object(torch.mps, "synchronize") as sync, caplog.at_level("INFO"):
        for prompt in ("job-a", "job-a", "job-b"):
            runtime.record_dispatch(prompt)
    assert runtime.native_calls == 3
    assert sync.call_count == 2
    assert caplog.text.count("Native INT8 active") == 2


def test_shader_is_ascii_for_pinned_pytorch_source_loader(worker):
    assert worker._SHADER.read_text(encoding="ascii").isascii()
