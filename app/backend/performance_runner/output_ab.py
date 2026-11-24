#!/usr/bin/env python3
"""Correctness A/B (cache OFF vs ON, fixed seed).

Proves the cache doesn't silently change output. Runs the SAME seed + prompt +
config twice — cache OFF then cache ON — extracts frames with ffmpeg and compares
them by per-frame PSNR.

On a deterministic path OFF vs ON match at very high PSNR; a meaningful divergence
means the cache handed back a module that differs from a fresh rebuild (a config the
cache key doesn't capture). Per the source review, ltx_pipelines fixes LoRAs at
construction, so this is expected to PASS on distilled/HQ/IC-LoRA — it's the
confirmation + the regression baseline for upstream rev bumps.

A cache correctness break would show as a large, obvious PSNR drop (wrong weights),
not a subtle fp difference — so decoded-frame PSNR is a sufficient gate here without
a latent-dump hook in the production pipeline.

Usage (backend running):
    python output_ab.py --seed 42 --run
    python output_ab.py --video-off a.mp4 --video-on b.mp4   # compare existing files
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile

import perf_config

# PSNR floor for a PASS. Identical decode -> inf; GPU/compile nondeterminism -> very
# high PSNR; a real divergence (wrong module) -> low.
PSNR_PASS_DB = 45.0


def _extract_frames(video: str, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video, os.path.join(out_dir, "f_%05d.png")],
        check=True,
    )
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png"))


def diff_videos(video_off: str, video_on: str) -> bool:
    """True = PASS (frames near-identical). Prints a summary + verdict."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        fo = _extract_frames(video_off, os.path.join(d, "off"))
        fn = _extract_frames(video_on, os.path.join(d, "on"))
        n = min(len(fo), len(fn))
        if n == 0:
            print("FAIL: no frames extracted")
            return False
        psnrs = []
        for i in range(n):
            a = np.asarray(Image.open(fo[i]).convert("RGB"), dtype="float64")
            b = np.asarray(Image.open(fn[i]).convert("RGB"), dtype="float64")
            if a.shape != b.shape:
                print(f"FAIL: frame {i} shape mismatch")
                return False
            mse = float(np.mean((a - b) ** 2))
            psnrs.append(99.0 if mse == 0 else 10 * np.log10((255.0 ** 2) / mse))
        ok = min(psnrs) >= PSNR_PASS_DB
        print("===== FRAME A/B =====")
        print(f"frames compared : {n}")
        print(f"PSNR min/median : {min(psnrs):.1f} / {statistics.median(psnrs):.1f} dB")
        print("VERDICT         :", f"PASS (near-identical, min >= {PSNR_PASS_DB:.0f} dB)" if ok else
              f"FAIL (frames diverge, min < {PSNR_PASS_DB:.0f} dB — cache handed back a different module)")
        return ok


def run_pair(seed: int, overrides: dict) -> tuple[str | None, str | None]:
    perf_config.wait_for_backend()
    ov = {**overrides, "seed": seed}
    perf_config.set_cache_enabled(False)
    off = perf_config.trigger_generation(ov)
    perf_config.set_cache_enabled(True)
    on = perf_config.trigger_generation(ov)
    print(f"[ab] off -> {off}\n[ab] on  -> {on}")
    return off, on


def main() -> int:
    """Returns a process exit code: 0 = PASS, 1 = FAIL/divergence, 2 = nothing to compare."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", action="store_true", help="trigger the OFF/ON pair via the backend")
    ap.add_argument("--video-off")
    ap.add_argument("--video-on")
    ap.add_argument("--overrides", default="{}")
    args = ap.parse_args()

    overrides = json.loads(args.overrides)

    off, on = args.video_off, args.video_on
    if args.run:
        off, on = run_pair(args.seed, overrides)
    if off and on:
        return 0 if diff_videos(off, on) else 1
    print("Nothing to compare. Use --run, or pass --video-off/--video-on.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
