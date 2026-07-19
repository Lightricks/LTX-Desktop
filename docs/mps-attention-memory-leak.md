# MPS attention memory leak — why bf16 generation OOMs on Apple Silicon, and the fix

## TL;DR

On Apple Silicon (MPS), local video generation grew GPU/driver memory unbounded
and OOM'd partway through — a 5 s / 121-frame job never completed on a 48 GB
M4 Pro. Root cause: **`mps-sdpa`'s fast zero-copy attention backend
(`mpsgraph_zc`) silently fails to build, so it falls back to the pure-pyobjc
`mpsgraph` backend, which leaks Metal memory on every attention call.** The
zero-copy backend doesn't leak — it just wasn't building because the `ninja`
binary wasn't on the backend's PATH. With `mpsgraph_zc` active, a full 5 s
generation runs with **flat ~5 GB** memory.

It is **not** related to fp8, the streaming/offload config, or anything in the
desktop wiring — all of those were ruled out with instrumentation.

## Symptom

- `runtime_policy` correctly selects `streaming_models_loading` on Darwin.
- Generation reaches the denoise loop, then `driver_allocated_memory` climbs
  ~2 GB / 15 s while live tensors (`current_allocated_memory`) stay flat at
  ~2.7 GB. `torch.mps.empty_cache()` reclaims ~nothing.
- Memory crosses 48 GB → macOS swaps → the process stalls → the Electron
  liveness probe gets no `/health` response for its window → backend killed and
  restarted. Looks like a "crash at stage 2"; it's actually an OOM stall.

## Investigation (what was ruled out)

| Hypothesis | Verdict |
|---|---|
| fp8 path leaks | ❌ bf16 leaks identically; fp8 isn't even used on MPS |
| streaming/offload mis-wired | ❌ `offload_mode_for_prefetch_count` correctly picks `OffloadMode.DISK` for MPS |
| `pinned_pool_fix` patch | ❌ dead no-op on this ltx-core rev (imports the old `layer_streaming`) |
| per-call autorelease-pool accumulation | ❌ wrapping each call in `objc.autorelease_pool()` did not flatten the curve |
| torch-native SDPA instead of mps-sdpa | ❌ hits the score-matrix memory wall at video sequence lengths |

Key signal: the leak is **sequence-length dependent**. At 49 frames (2 s), a
480×256 stage-1 held **flat** at ~5 GB and the full generation completed
(stage-2 upscale peaked ~29 GB). At 121 frames (5 s), stage-1 climbed unbounded.
So the pre-fix envelope on a 48 GB Mac was ~2 s.

## Root cause

`mps-sdpa` picks a backend in this order: `mpsgraph_zc` (zero-copy C++/Obj-C++
extension) → `mpsgraph` (pyobjc) → `metal_op` (unimplemented stub) → `stock`
(torch SDPA). On the affected machine the available list was
`['metal_proto', 'mpsgraph', 'stock']` — **`mpsgraph_zc` was missing**. Its
recorded reason:

```
extension build failed: RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)
```

`mpsgraph_zc` JIT-compiles its `.mm` via `torch.utils.cpp_extension.load()` on
first import, which needs the `ninja` binary. The `ninja` **Python package** was
present but its **binary wasn't on PATH** (launching `.venv/bin/python` directly
doesn't add the venv `bin/` to PATH). So `mpsgraph_zc` failed to build and
`mps-sdpa` fell back to the pyobjc `mpsgraph` backend.

The pyobjc backend (`backends/mpsgraph.py`) copies each Q/K/V/mask/out tensor
into a **freshly allocated `MTLBuffer` per attention call** (via CPU memcpy;
`_copy_tensor_to_tensor_data`). Those shared-storage Metal allocations accumulate
as `driver_allocated_memory` that torch's allocator can't reclaim — the observed
leak. (It's below the Python layer; per-call autorelease draining didn't help,
pointing at MPSGraph/Metal-internal retention triggered at longer sequences.)

## Fix

Get the **zero-copy `mpsgraph_zc`** backend to load — it manages buffers in C++
and doesn't do the per-call CPU-memcpy + fresh-`MTLBuffer` dance, so it doesn't
leak. Requirements:

1. `ninja` installed **and its binary on PATH** for the backend process.
2. A C++/Metal toolchain to compile it — **only at build time** (see production).

With `mpsgraph_zc` active, a 5 s / 121-frame generation holds **flat ~5 GB**.

### Changes landed in this repo (dev + build)

- `backend/pyproject.toml`: `ninja>=1.11; sys_platform == 'darwin'` as a dependency.
- `electron/python-backend.ts`: prepend the interpreter's `bin/` to the backend
  process PATH so torch can find `ninja`; pass `LTX_MPS_EXT_PREBUILT_DIR`.
- `backend/mps_prebuilt_ext.py`: on macOS, set `TORCH_EXTENSIONS_DIR` to a
  writable app-data dir and stage a bundled prebuilt cache into it (imported at
  backend startup, before `mps-sdpa`).
- `scripts/prebuild-mps-sdpa-ext.sh`: AOT-compile the extension against the
  shipped interpreter for bundling.

## Production shipping — the important caveat

`mps-sdpa` (upstream `crlandsc/mps-sdpa`) distributes **source-only** and
**JIT-compiles at runtime**, requiring **Xcode Command Line Tools** on the
machine. End users don't have those → on a user Mac the build fails → silent
fallback to the leaking backend → same OOM. **Bundling `ninja` alone is not
enough** (it's only the build orchestrator; the compile still needs clang + the
Metal SDK).

Proven approach (measured): a **prebuilt warm cache loads in ~1.7 s with only the
`ninja` binary present — clang is never invoked** (ninja sees the outputs
up-to-date and no-ops). So:

1. **Build time** (Mac CI `macos-latest`, has Xcode): `prepare-python.sh` Step 6.5
   runs `scripts/prebuild-mps-sdpa-ext.sh` → the cache is written **inside**
   `python-embed/mps-ext-prebuilt/mps_sdpa_zc_ext/`. It lives inside `python-embed`
   on purpose: CI caches `python-embed` (key includes `uv.lock` + `prepare-python.sh`)
   and skips the prepare step on a cache hit, so a cache that isn't part of
   `python-embed` would silently go missing on later builds.
2. **Bundle**: it ships with `python-embed` → `resources/python/mps-ext-prebuilt`;
   `ninja` ships inside `python-embed` too. `electron-builder.yml` `signIgnore`s the
   relocatable `.o` (codesign rejects it; it's never loaded — only the `.so` is).
3. **Runtime**: `mps_prebuilt_ext.py` copies it into the writable
   `TORCH_EXTENSIONS_DIR` on first run; torch loads it without a compiler.

Three things must be present in the shipped `python-embed` for this to work — all
now handled: (a) the **`ninja` binary** (dep + PATH); (b) **`setuptools`** —
`torch.utils.cpp_extension` imports it at module top, even to *load* a cached
extension, so `prepare-python.sh` no longer strips it on macOS; (c) the
**version-matched prebuilt cache** (rebuild whenever bundled torch/python changes).

Must-verify on the test release (**a clean, notarized install on a Mac with no
Xcode CLT**):
- `mpsgraph_zc` is in the available backends and clang is never invoked (ninja's
  up-to-date check is mtime-based; prebuilt outputs must stay newer than the
  bundled `.mm` — `mps_prebuilt_ext.py` touches them on stage to guarantee this).
- The staged `.so` loads under **hardened runtime**. It's signed with the app's
  team ID, so library validation should permit `dlopen` even from the
  user-writable `TORCH_EXTENSIONS_DIR` (validation keys on signature, not path).
  If it's blocked, add `com.apple.security.cs.disable-library-validation` to
  `resources/entitlements.mac.plist` (currently only `allow-jit`).
- Memory holds flat at ~5 GB for a 5 s generation (i.e. it didn't fall back).

Because option B has three moving parts in `python-embed` (ninja binary,
setuptools, version-matched cache) plus the signing nuance, the upstream
prebuilt-wheel (below) is the materially cleaner long-term fix.

### Recommended upstream ask (`crlandsc/mps-sdpa`)

Ship `mpsgraph_zc` as a **pre-compiled binary wheel** (a normal importable
extension, not torch JIT `load()`). Then no `ninja`, no clang, no Metal SDK at
runtime — the cleanest fix for every downstream consumer.

## Status / next steps

- [x] Root cause identified and fix validated in dev (flat memory at 5 s).
- [x] Dev/build enablement landed (ninja dep, PATH, runtime staging, prebuild script).
- [x] Prebuild wired into packaging (`prepare-python.sh` Step 6.5 → inside `python-embed`; `electron-builder.yml` signIgnore for the `.o`; keep setuptools on macOS).
- [x] `uv.lock` regenerated so `ninja` is locked.
- [ ] Test-release and verify on a clean, Xcode-less Mac (see "Must-verify" above).
- [ ] File the upstream request for prebuilt `mps-sdpa` wheels.
- [ ] (CI repo `Lightricks/ltx-desktop-ci`, optional) add `scripts/prebuild-mps-sdpa-ext.sh` to the `python-embed` cache-key `hashFiles` so edits to it also bust the cache. Not required now — the `uv.lock` + `prepare-python.sh` changes already bust it for this release.
