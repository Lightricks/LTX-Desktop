# ltx-kernels

Custom CUDA/C++ kernels for `ltx-core`. Four compiled extensions:

- **`all2all_cpp`** -- All2All communication kernels for multi-GPU tensor
  parallelism, used by the sequence-parallel inference path.
- **`ops_cpp`** -- Fused element ops for blockwise quantization: `rms_norm_rope`,
  `rms_norm_split_rope`, and FP6 pack/unpack.
- **`blockwise_cpp`** -- Blockwise FP8 GEMM. SM89 (GeForce/Ada) kernel always;
  the SM90 (Hopper, `deep_gemm`) kernel is added when a `9.0` architecture is
  requested.
- **`nvfp4_cpp`** -- NVFP4 (FP4 E2M1 + FP8 E4M3 per-16 block scales) quantize and
  cuBLASLt block-scaled GEMM. Built for Blackwell arches (`10.0` / `12.0`) that the
  local nvcc can emit (honors `TORCH_CUDA_ARCH_LIST` when set; skipped entirely if
  nvcc is too old). Needs a **Blackwell** GPU (SM ≥ 10.0) at runtime -- FP4 tensor
  cores do not exist before it. Python surface: `ltx_kernels.nvfp4`; see
  [`docs/NVFP4.md`](docs/NVFP4.md) for the layout contract (cuBLAS 128×4 block
  scales, Core42 PTQ checkpoint bytes). ltx-core drives it via `NVFP4Linear` and
  `build_nvfp4_*_policy`.

The Python surface for blockwise quantization lives in
`ltx_kernels.blockwise` (`functional`, `linear`, `triton_ops`).

`ltx_kernels.vae` adds two CuTe DSL kernels for the diffusion VAE decoder:

- **`na_attn_dsl`** -- standalone 3D neighborhood attention, a drop-in for
  `natten.na3d`, used by the decoder's deterministic stages.
- **`block_fna_dsl`** -- a whole `DiffusionNABlock` in one launch, with no full-volume
  Q/K/V, used by stage 5.

Production `sm_100a` wheels carry six CuTe AOT objects: plain and keyframe variants
for the two deterministic-stage kernels, plus plain and keyframe variants for the
LTX-2.5 stage-5 block. The launcher loads a matching object on first use, so a fresh
process does not compile those kernels. Non-production channel widths, custom kernel
or tile shapes, and the standalone context-only NA API keep the existing JIT fallback.

Both need a **datacenter Blackwell** GPU: they use `tcgen05` MMA *and* Tensor Memory
(`sm_100`/`sm_101`/`sm_103`). Consumer Blackwell has the former but not the latter, and
Hopper and Ada have neither, so this is not a slower fallback -- the instructions are
absent from those ISAs. Gate on `ltx_kernels.vae.block_fna_available` /
`na_attn_available`; each launcher also enforces it. ltx-core drives both through the
`NA_DSL_KERNELS` module op.

## Requirements

- CUDA toolkit (nvcc) matching your GPU architecture
- PyTorch with CUDA support
- Linux

## Building

`ltx-kernels` is excluded from the uv workspace, so a plain `uv sync` does not
build it. From the repository root, build it via the opt-in `kernels` group
(editable, no build isolation -- torch must already be installed):

```bash
uv sync --group kernels
```

Equivalently, install it directly:

```bash
uv pip install -e packages/ltx-kernels --no-build-isolation
```

Set `TORCH_CUDA_ARCH_LIST` to target specific architectures (speeds up compilation):

```bash
# H100 only
TORCH_CUDA_ARCH_LIST="9.0" uv pip install -e packages/ltx-kernels --no-build-isolation

# Multiple architectures
TORCH_CUDA_ARCH_LIST="9.0 9.0a 10.0 12.0" uv pip install -e packages/ltx-kernels --no-build-isolation
```

When `TORCH_CUDA_ARCH_LIST` is unset the build targets every supported
architecture (so `uv pip install` "just works" on a dev box); pin it on build
hosts to cut compile time. Any `9.0` entry enables the SM90 GEMM kernel, which
is compiled for `sm_90a` (the deep_gemm kernel uses wgmma/TMA).

To include the production VAE AOT bundle in a wheel, generate it in the package
tree before building. CuTe fake tensors make this an offline compile; the build
host does not need a GPU:

```bash
PYTHONPATH=packages/ltx-kernels/src python packages/ltx-kernels/scripts/build_vae_aot.py \
  --output-dir packages/ltx-kernels/src/ltx_kernels/vae/_aot/sm100a \
  --gpu-arch sm_100a --host-target linux-aarch64
```

CuTe's object-file format check decides whether the installed
`nvidia-cutlass-dsl` can load the object. Source/editable installs without a
bundle continue to JIT.

### cutlass headers

`blockwise_cpp` includes cute/cutlass headers (header-only; compiled into the
extension, with no runtime dependency). The build fetches them automatically on
first use: a blobless, `include/`-only sparse clone of cutlass pinned to commit
`afa17722` (v3.8.0), cached under `~/.cache/ltx-kernels/` (~25 MB) and reused
across builds.

- Set `CUTLASS_DIR=/path/to/cutlass` to use an existing checkout (uses
  `$CUTLASS_DIR/include` and skips the fetch).
- Set `LTX_KERNELS_CACHE_DIR` to override the cache location.

To bump cutlass, change `CUTLASS_REF` in `setup.py`.

## Testing

Tests require a CUDA GPU:

```bash
uv run pytest packages/ltx-kernels/tests/ -v
```

The `ltx_kernels.vae` and `ltx_kernels.nvfp4` tests additionally require a datacenter
Blackwell GPU and skip elsewhere. NVFP4 layout/API docs: [`docs/NVFP4.md`](docs/NVFP4.md).

```bash
uv run pytest packages/ltx-kernels/tests/ltx_kernels/test_nvfp4.py -v
```

## Operations

`all2all_cpp`:

- **send_recv_heads** -- Redistributes attention heads across GPUs (All2All)
- **gather_heads** -- Inverse of send_recv_heads
- **allgather** -- Gathers sequence tokens from all ranks

All operations support BFloat16 and Float8 (e4m3fn) data types.
