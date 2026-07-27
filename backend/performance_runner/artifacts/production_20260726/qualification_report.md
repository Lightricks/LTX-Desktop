# LTX Desktop Electron production qualification — 2026-07-26

Status: **PASS after dimension normalization fix `f539eb6`**

## Accepted production run

The final fixed-seed `fast_t2v_540p_5s` run used `POST /api/generate`, AUTO→MLX, BF16 eager mode, no tiling, and exact clean sidecar runtime `0.14.20.dev1` at `3171bac4ba901c0237faea2678c34034b37abc2a` with MLX 0.31.1.

| evidence | result |
|---|---:|
| completion-resolved dimensions | 896×512 |
| sidecar command dimensions | 896×512 |
| ffprobe dimensions | 896×512 |
| frames / duration | 121 / 5.0417 s |
| wall time | 66.608 s |
| peak process-tree RSS | 34.136 GiB |
| peak physical footprint | 41.133 GiB |
| peak MLX allocation | 38,585 MiB |
| peak GPU utilization sample | 100% |
| output SHA-256 | `5f1f02369e48d61b2951bc4d5b38321d4e6d22c65dee2e4870d6ee20de15185c` |

Strict failures are empty. The terminal runtime sample reports profile status `success`, active MLX memory 0 MiB, inactive allocator cache 2 MiB, MPS allocation 0 MiB, no active lease owner, and a free production flock. `useLocalTextEncoder` was restored to its original `false` value and the QA backend was stopped after the run.

## Pre-fix versus fixed run

| run | reported resolved | ffprobe actual | wall | physical | MLX peak | qualification |
|---|---|---|---:|---:|---:|---|
| pre-fix `c067989` | 960×544 | 960×512 | 76.013 s | 41.336 GiB | 38,679 MiB | reject: silent dimension mismatch |
| fixed `f539eb6` | 896×512 | 896×512 | 66.608 s | 41.133 GiB | 38,585 MiB | PASS |

The fixed run is 9.405 s faster, but it also renders 6.7% fewer pixels per frame; this is not treated as an optimization speedup claim. The important qualification change is that API, command, and media dimensions agree exactly on a valid distilled 64-pixel grid. The pre-fix result is retained as regression evidence even though its old harness status was PASS.

## Historical comparison

The preserved pre-change evidence is a direct-engine baseline, not an Electron app-entrypoint result. Its eager 704×448×96 run took 36.45 s with 35.20 GiB RSS and 38.81 GiB physical footprint; its eager 1280×704×96 run took 129.87 s with 35.20 GiB RSS and 45.42 GiB physical footprint. The final production run is about 1.20 seconds per million actual pixel-frames, comparable to the small historical run and below the historical HD value of about 1.50. Shape, frame count, entrypoint, and runtime orchestration differ, so these normalized figures are context rather than an A/B performance claim.

## Capability fallback, lease, and crash gates

- Runtime policy maps prepared text embeddings, A2V, Retake, Extend, IC-LoRA, and image generation to Torch. The interop smoke completed Electron job `de159ca4` through the prepared-embedding path while AI Studio waited for the shared lease. Its compact artifact proves completion and ordering, but does not persist a selected-engine field; it is routing/interop evidence, not a strict Torch performance measurement.
- Cross-product lease test: PASS. Electron owned first, AI Studio acquired only after Electron release, CPU-only control reads remained responsive, and the terminal flock was free.
- Crash release: PASS. Kernel flock authority released even when stale diagnostic JSON remained.
- TeaCache and explicit tiling: not applicable to Electron Fast distilled T2V/I2V. Dev/HQ TeaCache and modality-tiling candidates are covered only by the AI Studio matrix.

## Scoped Metal trace

The direct-engine Metal System Trace completed with target exit 0 and exact runtime identity. After PID filtering it contained 22,159 target GPU intervals; 22,149 were compute. Median/p95/p99/max interval durations were 1.384/7.951/9.122/33.928 ms. Shader Timeline was disabled, so the report makes command-buffer/encoder hotspot claims only, not kernel-symbol claims. The raw trace and large XML exports were not committed.

Compact strict results, interop/crash proofs, trace manifest/hotspots, and reports are retained. Raw process/progress/runtime samples were pruned after deriving the evidence above.
