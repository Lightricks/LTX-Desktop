#!/usr/bin/env python3
"""Decompose experiment — quantify how much wall time the cache patch saves.

Runs the same generation with the patch OFF then ON and reports median wall time
per config plus the delta:

    baseline    : patch OFF   (full disk read + fp8 + H2D + build on stage_2)
    patch_only  : patch ON    (stage_2 fully skipped on a cache hit)

Interpretation:
    baseline - patch_only  ~= total rebuild time the patch saves per generation

NOTE: this measures at the *generation* granularity (wall time). A finer
per-stage split (disk vs fp8 vs H2D) needs the in-process Step-0 instrumentation
(log torch.cuda.memory_allocated post-evict/post-build).

Usage (backend running on the remote 5090):
    python decompose.py --reps 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import perf_config

CONFIGS = {
    "baseline":   False,   # patch OFF
    "patch_only": True,    # patch ON
}


def main() -> int:
    """Measurement only (no pass/fail gate) — always exits 0 once it completes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5, help="repetitions per config (median-reported)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--warmup", type=int, default=1, help="discarded warmup gens per config")
    ap.add_argument("--overrides", default="{}")
    args = ap.parse_args()

    overrides = json.loads(args.overrides)

    perf_config.wait_for_backend()
    results: dict[str, list[float]] = {}

    for cfg in args.configs:
        patch = CONFIGS[cfg]
        perf_config.set_cache_enabled(patch)
        times: list[float] = []
        print(f"\n[decompose] config={cfg} (patch={patch})")
        for i in range(args.warmup + args.reps):
            t0 = time.time()
            perf_config.trigger_generation(overrides)
            wall = time.time() - t0
            tag = "warmup" if i < args.warmup else "measure"
            print(f"  {tag:>7} gen {i}: {wall:7.1f}s")
            if i >= args.warmup:
                times.append(wall)
        results[cfg] = times

    print("\n===== DECOMPOSE SUMMARY (median wall seconds) =====")
    med = {c: statistics.median(v) for c, v in results.items()}
    for c in args.configs:
        print(f"  {c:<12}: {med[c]:7.1f}s")

    if "baseline" in med and "patch_only" in med:
        saved = med["baseline"] - med["patch_only"]
        print(f"\n  rebuild time saved by patch  ~= baseline - patch_only = {saved:+.1f}s")
        if med["baseline"]:
            print(f"  as a fraction of baseline    ~= {saved / med['baseline'] * 100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
