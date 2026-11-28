# NVFP4 in `ltx-kernels`

In-house NVFP4 (FP4 E2M1 data + FP8 E4M3 per-16 block scales + FP32 per-tensor scale)
quantization and GEMM for LTX-2 Linear layers on Blackwell (SM ≥ 10.0).

Build: `TORCH_CUDA_ARCH_LIST=10.0 uv pip install -e packages/ltx-kernels --no-build-isolation`
(or `uv sync --group kernels` on a CUDA host). The `nvfp4_cpp` extension is skipped on
hosts without SM ≥ 10.

Post-productization follow-ups (CuTe GEMM, RMSNorm↔quant fusion, serving/CI):
[ENG-525](https://linear.app/lightricks/issue/ENG-525/nvfp4-ltx-kernels-follow-ups-post-productization).

## Format / layout contract

Per 2-D operand `x` of shape `(rows, K)`:

| piece | value | storage |
|---|---|---|
| per-tensor scale `s_t` | *decode* scale `amax / (E4M3_MAX * E2M1_MAX)` = `amax / 2688` | fp32 scalar |
| block scale `s_b[r, j]` | `e4m3_rne(clamp(blk_amax / 6 / s_t, max=448))` | `float8_e4m3fn`, cuBLAS 128×4 tiled |
| data | `e2m1_rne(x / (s_t * s_b))`, saturating at ±6 | `uint8`, two values per byte |

Reconstruction: `x ≈ data * s_b * s_t`.

### Nibble order (`hi_first`)

`hi_first=True` (default) puts element `2j` in the **high** nibble of byte `j`;
`hi_first=False` puts it in the low nibble. Pre-quantized checkpoints used with
`nvfp4-prequant` are expected in the default (`hi_first=True`) order.

Measured fact (see `tests/ltx_kernels/test_nvfp4.py::test_nibble_order_is_gemm_invariant`): the block
scale covers all 16 elements of a block, so swapping nibble order in **both** operands
only permutes the terms of each dot product — the result is bit-identical. A nibble swap
is only needed when *mixing* layouts.

### Block-scale swizzle

cuBLAS 1×16 block scaling wants the scale matrix `(rows, K/16)` zero-padded to
`(roundup(rows,128), roundup(K/16,4))` and shuffled into 128×4 tiles
([cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)).
Our kernel writes the swizzled address directly:

```
tile   = (row/128) * (padded_cols/4) + col/4
offset = tile*512 + (row%32)*16 + ((row%128)/32)*4 + col%4
```

### Padding rules

* `K % 32 == 0` is **required** (cuBLASLt FP4 constraint). LTX-2 DiT Linear `in_features`
  are multiples of 64.
* `N % 8 == 0` required for the GEMM (weight `out_features`).
* `M` (token count) is **unconstrained**: we do not pad the data, we only zero-pad the
  *scale* array up to 128 rows / 4 cols.

## GEMM strategy

**cuBLASLt FP4 block-scaled matmul.** `CUDA_R_4F_E2M1` operands, TN layout,
`A/B_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, `CUBLAS_COMPUTE_32F`,
bias fused via `CUBLASLT_EPILOGUE_BIAS`. Plan + heuristic are cached per
`(M,N,K,bias,out_dtype)`, and the per-tensor product `alpha` is a **device** scalar
(`CUBLASLT_POINTER_MODE_DEVICE`) so a dynamic activation scale never forces a `.item()`
sync or a second elementwise pass over the output.

## Compile / Dynamo

`ltx_kernels.nvfp4` exposes plain functions plus `torch.library.custom_op` wrappers
(`ltx_kernels_nvfp4::quantize` / `::scaled_mm` / `::amax_scale`) with `register_fake`
shape functions, so Dynamo treats them as opaque and
`CompilationConfig(seq_dim_dynamic=True, capture=True)` can CUDA-graph them.

* no host syncs (hence device `alpha`), no `.item()`, no data-dependent shapes;
* the cuBLASLt workspace is a per-device static tensor allocated on first use;
* the plan cache is keyed on shapes only, so capture replays hit the cache.

## Python API

```python
from ltx_kernels.nvfp4 import (
    is_available,
    linear_nvfp4,
    per_tensor_decode_scale,
    quantize_nvfp4,
    scaled_mm_nvfp4,
)

assert is_available()  # SM >= 10 and nvfp4_cpp built
decode = per_tensor_decode_scale(weight_bf16)
w_fp4, w_scale = quantize_nvfp4(weight_bf16, decode)
out = linear_nvfp4(x_bf16, w_fp4, w_scale, decode, bias=bias, act_scale=None)
```

`act_scale=None` uses a fixed per-tensor scale of `1.0` (no activation amax).
Pass a calibrated scalar (e.g. checkpoint `<layer>.input_scale`) for static
per-tensor activation decode scale; per-16 E4M3 block scales are still computed
each call. GEMM `alpha = act_scale * weight_scale`.
