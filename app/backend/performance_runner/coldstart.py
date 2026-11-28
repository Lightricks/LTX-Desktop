#!/usr/bin/env python3
"""Cold-start / first-generation latency.

On desktop the FIRST generation after a fresh backend launch pays a one-time cost
the user feels directly: model weights read from disk, moved to the GPU, and built.
This tool measures that penalty. Run it right after starting the backend (before any
other generation) for a true cold reading.

    python coldstart.py --warm 3

Reports:
    cold        : gen 0 wall time (includes the model load / disk read / build)
    warm median : median of the following gens (steady state)
    penalty     : cold - warm  (the one-time first-generation cost)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import perf_config


def main() -> int:
    """Exit code is always 0 — this is a measurement, not a pass/fail gate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=3, help="warm gens to run after the cold one")
    ap.add_argument("--overrides", default="{}", help="JSON overrides for the gen payload")
    args = ap.parse_args()
    overrides = json.loads(args.overrides)

    perf_config.wait_for_backend()
    print(f"[coldstart] request: POST {perf_config.GENERATE_PATH} "
          f"{json.dumps({**perf_config.GEN_PAYLOAD_TEMPLATE, **overrides})}")
    times: list[float] = []
    for i in range(1 + args.warm):
        t0 = time.time()
        perf_config.trigger_generation(overrides)
        wall = time.time() - t0
        print(f"[coldstart] gen {i} ({'cold' if i == 0 else 'warm'}): {wall:7.1f}s", flush=True)
        times.append(wall)

    cold = times[0]
    warm = statistics.median(times[1:]) if len(times) > 1 else cold
    penalty = cold - warm
    print("\n===== COLD-START SUMMARY =====")
    print(f"cold (gen 0)                    : {cold:.1f}s")
    print(f"warm median                     : {warm:.1f}s")
    print(f"first-gen penalty (cold - warm) : {penalty:+.1f}s")
    print("NOTE: run right after a fresh backend start for a true cold reading.")
    print(f"VERDICT: cold={cold:.1f}s  warm={warm:.1f}s  penalty={penalty:+.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
