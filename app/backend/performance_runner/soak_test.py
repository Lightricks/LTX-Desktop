#!/usr/bin/env python3
"""Compile-on soak (the primary leak gate).

Runs N generations back-to-back in one process and asserts the inter-generation
VRAM *floor* stays flat. A rising floor = the cache's `.to("meta")` evict isn't
actually reclaiming (compiled-module / cudagraph / lingering-reference leak).

Run this with Torch Compile ON (the axis that's cumulative and untested
single-shot). The leak gate is the VRAM floor; wall-time drift is advisory.

    python soak_test.py --n 50 --label cache_on --set-cache on
    python soak_test.py --n 50 --pair                 # cache_on + cache_off + compare

`--pair` runs both halves back-to-back in one process and checks that the cache_on
floor *tracks* the cache_off floor (so any drift is attributable to the patch, not
the pipeline). Emits soak_<label>_<ts>.csv per half and a PASS/FAIL summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time

import perf_config
from gpu_sampler import GpuSampler


def _slope(ys: list[float]) -> float:
    """Least-squares slope of ys vs index (MiB or s per generation)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def run_soak(*, n: int, label: str, set_cache: str, settle: float, overrides: dict, ts: str
             ) -> list[perf_config.RunResult]:
    """Run one N-generation soak, streaming rows to soak_<label>_<ts>.csv."""
    if set_cache != "skip":
        perf_config.set_cache_enabled(set_cache == "on")

    sampler = GpuSampler(hz=10).start()
    rows: list[perf_config.RunResult] = []
    # Incremental CSV (header up front, flush per gen): a stopped run keeps its data
    # and the dashboard chart updates live.
    csv_path = perf_config.RUNS_DIR / f"soak_{label}_{ts}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["gen", "wall_s", "peak_mib", "floor_mib", "ram_floor_mib",
                     "temp_c", "clock_mhz", "power_w"])
    csv_file.flush()

    print(f"[soak] label={label} n={n} cache={set_cache} baseline_vram={sampler.current()} MiB")
    print(f"[soak] {label}: csv -> {csv_path}")   # up front so the dashboard can chart THIS run live
    print(f"[soak] request: POST {perf_config.GENERATE_PATH} "
          f"{json.dumps({**perf_config.GEN_PAYLOAD_TEMPLATE, **overrides})}")
    try:
        for i in range(n):
            sampler.mark(f"g{i}_start")
            t0 = time.time()
            out = perf_config.trigger_generation(overrides)
            wall = time.time() - t0
            sampler.mark(f"g{i}_end")
            time.sleep(settle)  # let post-gen teardown settle before reading floor
            sampler.mark(f"g{i}_settled")

            peak = sampler.window_peak(since=f"g{i}_start", until=f"g{i}_end")
            floor = sampler.window_floor(since=f"g{i}_end", until=f"g{i}_settled")
            ram_floor = sampler.window_ram_floor(since=f"g{i}_end", until=f"g{i}_settled")
            temp = sampler.window_temp_max(since=f"g{i}_start", until=f"g{i}_end")
            clock = sampler.window_clock_mean(since=f"g{i}_start", until=f"g{i}_end")
            power = sampler.window_power_mean(since=f"g{i}_start", until=f"g{i}_end")
            r = perf_config.RunResult(label, i, wall, peak, floor, out, ram_floor, temp, clock, power)
            rows.append(r)
            fmt = lambda x: "" if x is None else f"{x:.0f}"  # noqa: E731
            writer.writerow([r.gen_index, f"{r.wall_s:.2f}", r.peak_mib, r.floor_mib,
                             fmt(ram_floor), fmt(temp), fmt(clock), fmt(power)])
            csv_file.flush()
            ram_str = "n/a" if ram_floor is None else f"{ram_floor:>7.0f} MiB"
            therm = "" if temp is None else f"  temp={temp:.0f}°C clock={clock:.0f}MHz"
            print(f"[soak] gen {i:>3}  wall={wall:7.1f}s  peak={peak:>6} MiB  floor={floor:>6} MiB  ram_floor={ram_str}{therm}")
    finally:
        sampler.stop()
        csv_file.close()
    print(f"[soak] {label}: csv -> {csv_path}")
    return rows


def summarize(rows: list[perf_config.RunResult], slope_ceiling: float, wall_drift: float,
              video_s: float = 0.0) -> dict:
    """Warm-window slopes + pass flags for a soak's rows (gen 0 excluded as cold-start)."""
    floors = [float(r.floor_mib) for r in rows]
    walls = [r.wall_s for r in rows]
    peaks = [r.peak_mib for r in rows]
    ram_floors = [float(r.ram_floor_mib) for r in rows if r.ram_floor_mib is not None]
    warm_floors = floors[1:] if len(floors) > 2 else floors
    warm_walls = walls[1:] if len(walls) > 2 else walls
    warm_ram = ram_floors[1:] if len(ram_floors) > 2 else ram_floors
    slope_warm = _slope(warm_floors)
    wall_slope = _slope(warm_walls)
    wall_median = statistics.median(walls) if walls else 0.0
    wall_slope_frac = (wall_slope / wall_median) if wall_median else 0.0
    ram_slope = _slope(warm_ram) if warm_ram else 0.0

    # Thermal / throttle: clock trending DOWN while the GPU is hot = throttling (a wall-time
    # drift that isn't a leak). Energy: mean power x wall, summed. Throughput: seconds of
    # video produced per second of wall time.
    warm_rows = rows[1:] if len(rows) > 2 else rows
    temps = [r.temp_c for r in rows if r.temp_c is not None]
    warm_clocks = [r.clock_mhz for r in warm_rows if r.clock_mhz is not None]
    clock_slope = _slope(warm_clocks) if len(warm_clocks) > 1 else 0.0
    temp_max = max(temps) if temps else None
    throttle = clock_slope < -5.0 and temp_max is not None and temp_max >= 80.0
    energy_wh = sum((r.power_w or 0.0) * r.wall_s for r in rows) / 3600.0
    peak_max = max(peaks) if peaks else 0
    total_mib = perf_config.gpu_total_mib()
    return {
        "floors": floors, "walls": walls, "peaks": peaks, "ram_floors": ram_floors,
        "slope_warm": slope_warm, "wall_slope": wall_slope, "ram_slope": ram_slope,
        "wall_slope_frac": wall_slope_frac, "wall_median": wall_median,
        "floor_ok": slope_warm <= slope_ceiling,
        "wall_ok": wall_slope_frac <= wall_drift,
        "ram_ok": ram_slope <= slope_ceiling,       # same MiB/gen ceiling, advisory only
        "temp_max": temp_max, "clock_slope": clock_slope, "throttle": throttle,
        "energy_wh": energy_wh, "video_s": video_s,
        "realtime": (video_s / wall_median) if (video_s and wall_median) else None,
        "peak_max": peak_max, "total_mib": total_mib,
        "headroom_pct": (100.0 * peak_max / total_mib) if (peak_max and total_mib) else None,
    }


def print_summary(label: str, s: dict, slope_ceiling: float, wall_drift: float) -> None:
    print(f"\n===== SOAK SUMMARY [{label}] =====")
    print(f"floor first/last   : {s['floors'][0]:.0f} -> {s['floors'][-1]:.0f} MiB")
    print(f"floor slope (warm) : {s['slope_warm']:+.2f} MiB/gen   (ceiling {slope_ceiling})   -> {'ok' if s['floor_ok'] else 'LEAK'}")
    print(f"wall first/last    : {s['walls'][0]:.1f} -> {s['walls'][-1]:.1f} s")
    print(f"wall slope (warm)  : {s['wall_slope']:+.3f} s/gen  ({s['wall_slope_frac']*100:+.2f}%/gen, advisory {wall_drift*100:.2f}%)   -> {'ok' if s['wall_ok'] else 'DRIFT'}")
    headroom = f"  ({s['headroom_pct']:.0f}% of {s['total_mib']} MiB)" if s.get("headroom_pct") else ""
    print(f"peak max           : {s['peak_max']} MiB   -> {perf_config.vram_fit(s['peak_max'])}{headroom}")
    if s["ram_floors"]:
        print(f"host RAM floor     : {s['ram_floors'][0]:.0f} -> {s['ram_floors'][-1]:.0f} MiB   "
              f"(slope {s['ram_slope']:+.2f} MiB/gen)   -> {'ok' if s['ram_ok'] else 'SPILL?'}")
    if s.get("realtime"):
        print(f"throughput         : {s['realtime']:.2f}x realtime ({s['video_s']:.0f}s video / {s['wall_median']:.1f}s wall median)")
    if s["temp_max"] is not None:
        print(f"GPU temp / clock   : max {s['temp_max']:.0f}°C · clock slope {s['clock_slope']:+.1f} MHz/gen   -> {'THROTTLING?' if s['throttle'] else 'ok'}")
    if s["energy_wh"]:
        per_gen_mwh = s["energy_wh"] / max(len(s["walls"]), 1) * 1000
        print(f"energy             : ~{s['energy_wh']:.1f} Wh total  (~{per_gen_mwh:.0f} mWh/gen)")
    print(f"wall median        : {s['wall_median']:.1f}s")
    # The leak gate is the VRAM FLOOR. Wall-time drift is advisory: on a contended box
    # it inflates with no leak, and can't be told from a real spill by wall time alone.
    if not s["wall_ok"]:
        print("ADVISORY           : wall-time drifting while the floor is",
              "flat" if s["floor_ok"] else "also rising", "— re-run on an IDLE box")
        print("                     (no other GPU users); if it persists, suspect a spill.")
    if s["ram_floors"] and not s["ram_ok"]:
        print("ADVISORY           : host-RAM floor rising across gens — possible leak/spill into")
        print("                     system RAM (affects the whole machine, not just VRAM).")


def main() -> int:
    """Returns a process exit code: 0 = PASS (floor flat / tracks control), 1 = FAIL."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of generations")
    ap.add_argument("--label", default="run")
    ap.add_argument("--pair", action="store_true", help="run cache_on + cache_off and compare")
    ap.add_argument("--settle", type=float, default=4.0, help="settle seconds after each gen")
    ap.add_argument("--set-cache", choices=["on", "off", "skip"], default="skip",
                    help="set the patch toggle before a single run ('skip' leaves it as-is)")
    ap.add_argument("--slope-ceiling-mib", type=float, default=50.0, help="max floor slope MiB/gen")
    ap.add_argument("--wall-drift-frac", type=float, default=0.005, help="advisory wall-drift ceiling")
    ap.add_argument("--track-margin-mib", type=float, default=30.0,
                    help="--pair: max MiB/gen the cache_on floor may drift ABOVE cache_off")
    ap.add_argument("--no-idle-check", action="store_true", help="skip the pre-run GPU-idle check")
    ap.add_argument("--overrides", default="{}", help="JSON overrides for the gen payload")
    args = ap.parse_args()

    overrides = json.loads(args.overrides)
    video_s = float({**perf_config.GEN_PAYLOAD_TEMPLATE, **overrides}.get("duration", 0) or 0)
    perf_config.wait_for_backend()

    if not args.no_idle_check:
        busy, util = perf_config.gpu_busy()
        if busy:
            print(f"[soak] WARNING: GPU already ~{util:.0f}% busy — another process is using it.")
            print("[soak]          Wall-time will be contaminated; close other GPU apps for a clean run.\n")

    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.pair:
        on = summarize(run_soak(n=args.n, label="cache_on", set_cache="on", settle=args.settle,
                                overrides=overrides, ts=ts), args.slope_ceiling_mib, args.wall_drift_frac, video_s)
        off = summarize(run_soak(n=args.n, label="cache_off", set_cache="off", settle=args.settle,
                                 overrides=overrides, ts=ts), args.slope_ceiling_mib, args.wall_drift_frac, video_s)
        print_summary("cache_on", on, args.slope_ceiling_mib, args.wall_drift_frac)
        print_summary("cache_off", off, args.slope_ceiling_mib, args.wall_drift_frac)

        diff = on["slope_warm"] - off["slope_warm"]
        tracks = diff <= args.track_margin_mib
        print("\n===== PAIR COMPARISON =====")
        print(f"cache_on  warm floor slope : {on['slope_warm']:+.2f} MiB/gen")
        print(f"cache_off warm floor slope : {off['slope_warm']:+.2f} MiB/gen")
        print(f"on - off                   : {diff:+.2f} MiB/gen   (track margin {args.track_margin_mib})")
        if not on["floor_ok"]:
            verdict = "FAIL (cache_on floor rising beyond ceiling -> leak)"
        elif not tracks:
            verdict = f"FAIL (cache_on drifts +{diff:.0f} MiB/gen above cache_off -> patch adds memory growth)"
        elif not off["floor_ok"]:
            verdict = "PASS (cache_on ok; cache_off also drifts -> environmental, not the patch)"
        else:
            verdict = "PASS (cache_on floor tracks cache_off -> no patch-induced leak)"
        print(f"VERDICT                    : {verdict}")
        return 0 if verdict.startswith("PASS") else 1

    rows = run_soak(n=args.n, label=args.label, set_cache=args.set_cache, settle=args.settle,
                    overrides=overrides, ts=ts)
    if not rows:
        print("[soak] no generations completed — nothing to summarize")
        return 1
    s = summarize(rows, args.slope_ceiling_mib, args.wall_drift_frac, video_s)
    print_summary(args.label, s, args.slope_ceiling_mib, args.wall_drift_frac)
    if s["floor_ok"]:
        verdict = "PASS (floor flat -> no leak)" + ("" if s["wall_ok"] else "  [wall-drift advisory, see above]")
    else:
        verdict = "FAIL (rising VRAM floor -> evict not reclaiming)"
    print(f"VERDICT            : {verdict}")
    return 0 if s["floor_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
