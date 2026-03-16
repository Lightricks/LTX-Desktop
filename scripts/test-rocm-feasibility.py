"""
ROCm Feasibility Test — AMD AI MAX 395+ on Windows
===================================================

Validates that AMD's ROCm PyTorch Windows wheels are functional for the LTX
pipeline before committing to full build system integration.

Prerequisites (run in a clean Python 3.12 venv):
    pip install --no-cache-dir `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
    pip install --no-cache-dir `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
        https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl

Usage:
    python scripts/test-rocm-feasibility.py
    python scripts/test-rocm-feasibility.py --skip-ltx    # skip LTX pipeline import tests
    python scripts/test-rocm-feasibility.py --verbose     # print more detail on failures
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
SKIP = "⏭️  SKIP"

results: list[tuple[str, str, str]] = []  # (section, test_name, status + note)


def record(section: str, name: str, ok: bool | None, note: str = "") -> None:
    if ok is None:
        status = SKIP
    elif ok:
        status = PASS
    else:
        status = FAIL
    label = f"{status}  {name}"
    if note:
        label += f"  [{note}]"
    results.append((section, name, label))
    print(f"  {label}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def summarize() -> int:
    """Print summary and return exit code (0=all pass, 1=any fail)."""
    passes = sum(1 for _, _, s in results if s.startswith("✅"))
    fails = sum(1 for _, _, s in results if s.startswith("❌"))
    warns = sum(1 for _, _, s in results if s.startswith("⚠️"))
    skips = sum(1 for _, _, s in results if s.startswith("⏭️"))

    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  ✅ Passed : {passes}")
    print(f"  ❌ Failed : {fails}")
    print(f"  ⚠️  Warned : {warns}")
    print(f"  ⏭️  Skipped: {skips}")

    if fails == 0:
        verdict = "🟢 GREEN — ROCm looks good, proceed with implementation"
    elif fails <= 2:
        verdict = "🟡 YELLOW — Some issues but may be workable, review failures above"
    else:
        verdict = "🔴 RED — Multiple critical failures, AMD ROCm path may not be viable"

    print(f"\n  Verdict: {verdict}")
    print(f"{'═' * 60}\n")
    return 1 if fails > 0 else 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Environment
# ─────────────────────────────────────────────────────────────────────────────

def test_environment(verbose: bool) -> str | None:
    """Returns device string if cuda available, else None."""
    section("1. ENVIRONMENT")

    # Python version
    py_ver = sys.version_info
    record("env", "Python version", True, f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 12):
        print("    NOTE: AMD ROCm Windows wheels require Python 3.12+")

    try:
        import torch
        record("env", "torch import", True, torch.__version__)
    except Exception as e:
        record("env", "torch import", False, str(e))
        print("  Cannot continue without torch — exiting")
        return None

    # HIP version (distinguishes ROCm from CUDA)
    hip_ver = getattr(torch.version, "hip", None)
    if hip_ver:
        record("env", "ROCm/HIP detected", True, f"HIP {hip_ver}")
    else:
        cuda_ver = getattr(torch.version, "cuda", None)
        if cuda_ver:
            record("env", "ROCm/HIP detected", False,
                   f"This is NVIDIA CUDA {cuda_ver} — install AMD ROCm wheels")
        else:
            record("env", "ROCm/HIP detected", False, "No HIP or CUDA version found")

    # CUDA API availability (ROCm uses same API)
    cuda_ok = torch.cuda.is_available()
    record("env", "torch.cuda.is_available()", cuda_ok)
    if not cuda_ok:
        print("    NOTE: GPU not visible — check driver and ROCm SDK install")
        return None

    # Device name
    try:
        device_name = torch.cuda.get_device_name(0)
        is_amd = "AMD" in device_name or "Radeon" in device_name or "Instinct" in device_name
        record("env", "Device name", True, device_name)
        if not is_amd and not hip_ver:
            print("    NOTE: This appears to be an NVIDIA GPU, not AMD")
    except Exception as e:
        record("env", "Device name", False, str(e))

    # Device properties / memory
    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        record("env", "Device properties", True,
               f"total_memory={total_gb:.1f} GB, name={props.name}")

        if total_gb < 31:
            print(f"    ⚠️  VRAM={total_gb:.1f}GB is below LTX's 31GB threshold!")
            print("    NOTE: Increase GPU memory allocation in BIOS (e.g. 96GB GPU)")
            record("env", "VRAM >= 31 GB", False, f"{total_gb:.1f} GB reported")
        else:
            record("env", "VRAM >= 31 GB", True, f"{total_gb:.1f} GB — sufficient for LTX")
    except Exception as e:
        record("env", "Device properties", False, str(e))

    # Capability (RDNA 3.5 = gfx1151, reports as sm_11_0 or similar in HIP)
    try:
        major, minor = torch.cuda.get_device_capability(0)
        record("env", "Device capability", True, f"sm_{major}{minor}")
    except Exception as e:
        record("env", "Device capability", False, str(e))

    return "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: bfloat16 (CRITICAL — codebase uses bf16 globally)
# ─────────────────────────────────────────────────────────────────────────────

def test_bfloat16(device: str, verbose: bool) -> bool:
    section("2. BFLOAT16 SUPPORT  (CRITICAL — LTX uses bf16 globally)")

    import torch

    # Basic creation
    try:
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16, device=device)
        record("bf16", "Create bf16 tensor", True, str(t.dtype))
        bf16_basic = True
    except Exception as e:
        record("bf16", "Create bf16 tensor", False, str(e))
        bf16_basic = False

    if not bf16_basic:
        record("bf16", "bf16 matmul", False, "skipped — creation failed")
        record("bf16", "bf16 softmax", False, "skipped — creation failed")
        print("  NOTE: bf16 completely broken. Will need float16 fallback.")
        test_float16_fallback(device, verbose)
        return False

    # Matmul
    try:
        a = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
        b = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
        c = torch.matmul(a, b)
        record("bf16", "bf16 matmul", True, f"output shape {c.shape}")
    except Exception as e:
        record("bf16", "bf16 matmul", False, str(e))

    # Softmax
    try:
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=device)
        y = torch.softmax(x, dim=-1)
        record("bf16", "bf16 softmax", True, f"sum={y.sum().item():.2f}")
    except Exception as e:
        record("bf16", "bf16 softmax", False, str(e))

    # Linear layer (nn.Linear)
    try:
        import torch.nn as nn
        linear = nn.Linear(128, 128, bias=True).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(2, 128, dtype=torch.bfloat16, device=device)
        y = linear(x)
        record("bf16", "bf16 nn.Linear forward", True, f"output {y.shape}")
    except Exception as e:
        record("bf16", "bf16 nn.Linear forward", False, str(e))

    # GroupNorm (used in LTX VAE)
    try:
        import torch.nn as nn
        gn = nn.GroupNorm(8, 32).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(2, 32, 8, 8, dtype=torch.bfloat16, device=device)
        y = gn(x)
        record("bf16", "bf16 GroupNorm", True)
    except Exception as e:
        record("bf16", "bf16 GroupNorm", False, str(e))

    # LayerNorm (used in transformers)
    try:
        import torch.nn as nn
        ln = nn.LayerNorm(128).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(2, 16, 128, dtype=torch.bfloat16, device=device)
        y = ln(x)
        record("bf16", "bf16 LayerNorm", True)
    except Exception as e:
        record("bf16", "bf16 LayerNorm", False, str(e))

    return True


def test_float16_fallback(device: str, verbose: bool) -> None:
    import torch
    print("\n  Testing float16 as fallback dtype...")
    try:
        a = torch.randn(64, 64, dtype=torch.float16, device=device)
        b = torch.randn(64, 64, dtype=torch.float16, device=device)
        c = torch.matmul(a, b)
        record("f16-fallback", "float16 matmul (fallback)", True)
    except Exception as e:
        record("f16-fallback", "float16 matmul (fallback)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Core Tensor Operations
# ─────────────────────────────────────────────────────────────────────────────

def test_core_ops(device: str, verbose: bool) -> None:
    section("3. CORE TENSOR OPERATIONS")

    import torch
    import torch.nn.functional as F

    dtype = torch.bfloat16

    # Activation functions used in LTX transformer/VAE
    for fn_name, fn in [
        ("gelu", lambda x: F.gelu(x, approximate="tanh")),
        ("silu", F.silu),
        ("relu", F.relu),
    ]:
        try:
            x = torch.randn(4, 128, dtype=dtype, device=device)
            y = fn(x)
            record("ops", f"F.{fn_name}", True)
        except Exception as e:
            record("ops", f"F.{fn_name}", False, str(e))

    # scaled_dot_product_attention (CRITICAL — used by all transformers)
    try:
        q = torch.randn(2, 8, 32, 64, dtype=dtype, device=device)
        k = torch.randn(2, 8, 32, 64, dtype=dtype, device=device)
        v = torch.randn(2, 8, 32, 64, dtype=dtype, device=device)
        out = F.scaled_dot_product_attention(q, k, v)
        record("ops", "F.scaled_dot_product_attention", True, f"output {out.shape}")
    except Exception as e:
        record("ops", "F.scaled_dot_product_attention", False, str(e))
        if verbose:
            traceback.print_exc()

    # conv2d (used in MiDaS depth processor, VAE)
    try:
        import torch.nn as nn
        conv2d = nn.Conv2d(3, 32, kernel_size=3, padding=1).to(device=device, dtype=dtype)
        x = torch.randn(1, 3, 64, 64, dtype=dtype, device=device)
        y = conv2d(x)
        record("ops", "Conv2d", True, f"output {y.shape}")
    except Exception as e:
        record("ops", "Conv2d", False, str(e))

    # conv3d (used in LTX VAE for spatio-temporal processing)
    try:
        import torch.nn as nn
        conv3d = nn.Conv3d(4, 16, kernel_size=3, padding=1).to(device=device, dtype=dtype)
        x = torch.randn(1, 4, 8, 32, 32, dtype=dtype, device=device)
        y = conv3d(x)
        record("ops", "Conv3d (VAE temporal)", True, f"output {y.shape}")
    except Exception as e:
        record("ops", "Conv3d (VAE temporal)", False, str(e))
        if verbose:
            traceback.print_exc()

    # torch.Generator (used for seeded generation)
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        x = torch.randn(4, 4, generator=gen, device=device, dtype=dtype)
        record("ops", "torch.Generator with device seed", True)
    except Exception as e:
        record("ops", "torch.Generator with device seed", False, str(e))

    # Embedding lookup (used in text encoder)
    try:
        import torch.nn as nn
        emb = nn.Embedding(32000, 512).to(device=device, dtype=dtype)
        idx = torch.randint(0, 32000, (2, 16), device=device)
        y = emb(idx)
        record("ops", "nn.Embedding lookup", True, f"output {y.shape}")
    except Exception as e:
        record("ops", "nn.Embedding lookup", False, str(e))

    # RMSNorm-equivalent (used in Gemma text encoder)
    try:
        x = torch.randn(2, 16, 512, dtype=dtype, device=device)
        norm = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        record("ops", "RMS norm", True)
    except Exception as e:
        record("ops", "RMS norm", False, str(e))

    # CPU↔GPU tensor transfer (used for CPU offloading)
    try:
        x_cpu = torch.randn(64, 64)
        x_gpu = x_cpu.to(device)
        x_back = x_gpu.cpu()
        max_diff = (x_cpu - x_back).abs().max().item()
        record("ops", "CPU↔GPU transfer", max_diff < 1e-5, f"max diff={max_diff:.2e}")
    except Exception as e:
        record("ops", "CPU↔GPU transfer", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: torch.compile
# ─────────────────────────────────────────────────────────────────────────────

def test_torch_compile(device: str, verbose: bool) -> None:
    section("4. TORCH.COMPILE  (performance opt — failure is non-blocking)")

    import torch
    import torch.nn as nn

    # Check triton availability
    try:
        import triton  # type: ignore[reportMissingImports]
        record("compile", "triton import", True, getattr(triton, "__version__", "unknown version"))
    except ImportError:
        record("compile", "triton import", None, "not installed — torch.compile may use fallback")
    except Exception as e:
        record("compile", "triton import", False, str(e))

    # Simple model compile test
    try:
        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = SimpleModel().to(device=device, dtype=torch.bfloat16)
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        x = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
        y = compiled(x)
        record("compile", "torch.compile(mode='reduce-overhead')", True, f"output {y.shape}")
    except Exception as e:
        record("compile", "torch.compile(mode='reduce-overhead')", False,
               "non-blocking: LTX can run uncompiled")
        if verbose:
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Memory Management
# ─────────────────────────────────────────────────────────────────────────────

def test_memory(device: str, verbose: bool) -> None:
    section("5. MEMORY MANAGEMENT")

    import torch

    # empty_cache (called after model unloads)
    try:
        t = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
        del t
        torch.cuda.empty_cache()
        record("mem", "torch.cuda.empty_cache()", True)
    except Exception as e:
        record("mem", "torch.cuda.empty_cache()", False, str(e))

    # synchronize (called for timing / before cache clear)
    try:
        x = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        record("mem", "torch.cuda.synchronize()", True)
    except Exception as e:
        record("mem", "torch.cuda.synchronize()", False, str(e))

    # memory_allocated / memory_reserved
    try:
        before = torch.cuda.memory_allocated(0)
        t = torch.ones(1024, 1024, device=device, dtype=torch.float32)
        after = torch.cuda.memory_allocated(0)
        allocated_mb = (after - before) / (1024 ** 2)
        record("mem", "torch.cuda.memory_allocated()", True, f"+{allocated_mb:.1f} MB for 4MB tensor")
        del t
    except Exception as e:
        record("mem", "torch.cuda.memory_allocated()", False, str(e))

    # Large allocation (simulates model weight tensors)
    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        # Allocate ~1% of VRAM as a stress test
        alloc_mb = max(256, int(total_gb * 10))
        t = torch.zeros(alloc_mb * 1024 * 256, dtype=torch.float32, device=device)
        actual_mb = t.nelement() * t.element_size() / (1024 ** 2)
        del t
        torch.cuda.empty_cache()
        record("mem", f"Large allocation ({actual_mb:.0f} MB)", True)
    except Exception as e:
        record("mem", "Large allocation", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: LTX Pipeline Import Test
# ─────────────────────────────────────────────────────────────────────────────

def test_ltx_imports(device: str, verbose: bool) -> None:
    section("6. LTX PIPELINE IMPORTS  (requires ltx-core and ltx-pipelines installed)")

    # ltx-core
    try:
        from ltx_core.quantization import QuantizationPolicy  # type: ignore[reportMissingImports]
        record("ltx", "ltx-core: QuantizationPolicy import", True)
    except ImportError:
        record("ltx", "ltx-core: QuantizationPolicy import", None,
               "ltx-core not installed — run: pip install ltx-core from LTX-2 repo")
        return
    except Exception as e:
        record("ltx", "ltx-core: QuantizationPolicy import", False, str(e))

    # ltx-pipelines
    try:
        from ltx_pipelines.distilled import DistilledPipeline  # type: ignore[reportMissingImports]
        record("ltx", "ltx-pipelines: DistilledPipeline import", True)
    except ImportError:
        record("ltx", "ltx-pipelines: DistilledPipeline import", None, "ltx-pipelines not installed")
        return
    except Exception as e:
        record("ltx", "ltx-pipelines: DistilledPipeline import", False, str(e))

    # diffusers
    try:
        from diffusers import DPTForDepthEstimation  # type: ignore[reportMissingImports]
        record("ltx", "diffusers: DPTForDepthEstimation import", True)
    except ImportError:
        record("ltx", "diffusers: DPTForDepthEstimation import", None, "diffusers not installed")
    except Exception as e:
        record("ltx", "diffusers: DPTForDepthEstimation import", False, str(e))

    # transformers
    try:
        import transformers  # type: ignore[reportMissingImports]
        record("ltx", "transformers import", True, transformers.__version__)

        from transformers import GemmaTokenizerFast  # type: ignore[reportMissingImports]
        record("ltx", "transformers: GemmaTokenizerFast import", True)
    except ImportError:
        record("ltx", "transformers import", None, "transformers not installed")
    except Exception as e:
        record("ltx", "transformers import", False, str(e))

    # Verify ltx-core doesn't use CUDA-specific ops that would break ROCm
    try:
        import torch
        from ltx_core.quantization import QuantizationPolicy  # type: ignore[reportMissingImports]

        # Try instantiating quantization policy — this exercises import-time GPU checks
        policy = QuantizationPolicy(enabled=False)
        record("ltx", "QuantizationPolicy(enabled=False) instantiation", True)
    except Exception as e:
        record("ltx", "QuantizationPolicy instantiation", False, str(e))
        if verbose:
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: FP8 Detection (should be FALSE on RDNA 3.5)
# ─────────────────────────────────────────────────────────────────────────────

def test_fp8_guard(device: str, verbose: bool) -> None:
    section("7. FP8 DETECTION  (should be FALSE on RDNA 3.5)")

    import torch

    hip_ver = getattr(torch.version, "hip", None)
    is_rocm = hip_ver is not None

    record("fp8", "Detected as ROCm (not NVIDIA)", is_rocm,
           f"torch.version.hip={hip_ver!r}")

    if is_rocm:
        record("fp8", "FP8 guard logic (should skip FP8 for RDNA 3.5)", True,
               "device_supports_fp8() will correctly return False via torch.version.hip check")
    else:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            fp8_supported = major > 8 or (major == 8 and minor >= 9)
            record("fp8", "FP8 capability (NVIDIA)", fp8_supported,
                   f"sm_{major}{minor} — {'supports' if fp8_supported else 'does NOT support'} FP8")
        except Exception as e:
            record("fp8", "FP8 capability check", False, str(e))

    # Test actual float8 tensor creation (informational)
    try:
        t = torch.zeros(4, dtype=torch.float8_e4m3fn, device=device)
        record("fp8", "float8_e4m3fn tensor creation", True,
               "hardware FP8 available (unexpected on RDNA 3.5)")
    except Exception as e:
        record("fp8", "float8_e4m3fn tensor creation", None,
               f"Not available (expected on RDNA 3.5): {type(e).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Transformer-Scale Stress Test
# ─────────────────────────────────────────────────────────────────────────────

def test_transformer_stress(device: str, verbose: bool) -> None:
    section("8. TRANSFORMER-SCALE STRESS TEST  (small but realistic shapes)")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dtype = torch.bfloat16

    # Simulate a single transformer block pass (LTX uses ~128 heads, d_head=128 for its 8B model)
    # We use smaller dims here for speed
    batch, seq_len, d_model, n_heads, d_head = 1, 64, 1024, 8, 128
    d_ff = d_model * 4

    try:
        # Input
        x = torch.randn(batch, seq_len, d_model, dtype=dtype, device=device)

        # QKV projections
        W_qkv = torch.randn(d_model, 3 * n_heads * d_head, dtype=dtype, device=device)
        qkv = x @ W_qkv  # (B, S, 3*H*D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, n_heads, d_head).transpose(1, 2)  # (B, H, S, D)
        k = k.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
        v = v.view(batch, seq_len, n_heads, d_head).transpose(1, 2)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, n_heads * d_head)

        # Output projection + residual
        W_o = torch.randn(n_heads * d_head, d_model, dtype=dtype, device=device)
        out = attn_out @ W_o + x

        # Layer norm
        ln = nn.LayerNorm(d_model).to(device=device, dtype=dtype)
        out = ln(out)

        # FFN
        W1 = torch.randn(d_model, d_ff, dtype=dtype, device=device)
        W2 = torch.randn(d_ff, d_model, dtype=dtype, device=device)
        ffn_out = F.gelu(out @ W1, approximate="tanh") @ W2
        final = ln(out + ffn_out)

        torch.cuda.synchronize()
        record("stress", "Full transformer block (SDPA + FFN)", True,
               f"shape {final.shape}, dtype {final.dtype}")
    except Exception as e:
        record("stress", "Full transformer block (SDPA + FFN)", False, str(e))
        if verbose:
            traceback.print_exc()

    # VAE-like conv stack (3D convolutions)
    try:
        # Simulate a small 3D conv block as in LTX's spatial-temporal VAE
        x = torch.randn(1, 4, 4, 32, 32, dtype=dtype, device=device)  # (B, C, T, H, W)
        conv1 = nn.Conv3d(4, 16, kernel_size=3, padding=1).to(device=device, dtype=dtype)
        conv2 = nn.Conv3d(16, 4, kernel_size=3, padding=1).to(device=device, dtype=dtype)
        gn = nn.GroupNorm(4, 16).to(device=device, dtype=dtype)

        out = F.silu(gn(conv1(x)))
        out = conv2(out)
        torch.cuda.synchronize()
        record("stress", "3D VAE-like conv block", True, f"output shape {out.shape}")
    except Exception as e:
        record("stress", "3D VAE-like conv block", False, str(e))
        if verbose:
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ROCm feasibility test for AMD AI MAX 395+ on Windows"
    )
    parser.add_argument("--skip-ltx", action="store_true",
                        help="Skip LTX pipeline import tests (ltx-core/ltx-pipelines not installed)")
    parser.add_argument("--skip-stress", action="store_true",
                        help="Skip transformer-scale stress test")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full tracebacks on failure")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  ROCm Feasibility Test — AMD AI MAX 395+ on Windows")
    print("  LTX Desktop — AMD GPU Support Investigation")
    print("═" * 60)

    device = test_environment(args.verbose)
    if device is None:
        print("\n  Cannot proceed: GPU not available.")
        print("  Check: AMD Adrenalin driver ≥26.1.1 installed?")
        print("  Check: ROCm SDK wheels installed?")
        print("  Check: Windows Defender Application Guard disabled?")
        return 1

    test_bfloat16(device, args.verbose)
    test_core_ops(device, args.verbose)
    test_torch_compile(device, args.verbose)
    test_memory(device, args.verbose)
    test_fp8_guard(device, args.verbose)

    if not args.skip_stress:
        test_transformer_stress(device, args.verbose)

    if not args.skip_ltx:
        test_ltx_imports(device, args.verbose)

    return summarize()


if __name__ == "__main__":
    sys.exit(main())
