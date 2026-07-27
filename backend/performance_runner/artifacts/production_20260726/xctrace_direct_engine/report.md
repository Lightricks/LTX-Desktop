# Scoped Metal System Trace qualification

Status: **PASS with symbol-level limitation**

The recorder launched one fixed-seed direct-engine workload under the `Metal System Trace` template while a wrapper held the production shared flock for the full subprocess lifetime. The target exited 0; xctrace record and TOC export both exited 0. The production Electron and AI Studio entrypoints were idle for the capture.

## Workload and identity

- Runtime: `ltx-pipelines-mlx 0.14.20.dev1` at clean revision `3171bac4ba901c0237faea2678c34034b37abc2a`
- Device: Apple M5 Max, 128 GiB unified memory, MLX 0.31.1
- Model: `dgrauet/ltx-2.3-mlx`, BF16, distilled two-stage
- Shape: 768×512×129 at 24 fps; seed 424242
- Target child: `python3.11` PID 54125, exit 0
- Engine-reported render time: 59.0 s
- Target GPU interval window: 56.439 s

## Per-dispatch hotspot summary

The focused export contains 22,159 target-process GPU intervals after excluding Codex, loginwindow, and other system Metal traffic. Compute accounts for 22,149 intervals and 52.298 s of the 52.300 s summed interval duration. Interval durations are nested and therefore are not additive GPU wall time.

| statistic | duration |
|---|---:|
| median | 1.384 ms |
| p95 | 7.951 ms |
| p99 | 9.122 ms |
| maximum | 33.928 ms |

The slowest recorded intervals were all compute encoders:

| trace start | duration | command buffer | encoder | submission |
|---:|---:|---|---|---:|
| 58.478 s | 33.928 ms | `0x60301691c` | `0x60301691d` | 67,265,732 |
| 3.845 s | 32.952 ms | `0x60300a907` | `0x60300a908` | 67,155,922 |
| 58.086 s | 25.967 ms | `0x6030168fb` | `0x6030168fc` | 67,265,393 |
| 57.886 s | 25.947 ms | `0x6030168e9` | `0x6030168ea` | 67,265,218 |
| 58.886 s | 25.929 ms | `0x60301693e` | `0x60301693f` | 67,266,092 |

`Compute Command 0` dominates the generic encoder labels (20,804 intervals; 51.280 summed seconds). The template recorded `Shader Timeline: Disabled`, so this capture cannot attribute those dispatches to kernel symbols or shader-counter metrics. The command-buffer/encoder IDs above are the most precise per-dispatch attribution supported by this capture; kernel-name claims would be unsupported.

## Artifact policy

`manifest_rerun.json`, `hotspots.json`, and this report are the compact committed evidence. The raw trace bundle remains at `/private/tmp/ltx_qa_20260726_rerun.trace`; focused XML exports are intentionally uncommitted because they are large derived data. The manifest does not contain a raw trace SHA because an `.trace` is a directory bundle rather than a regular file.
