#!/usr/bin/env python3
"""Sanity sweep — run Desktop-feature scenarios and produce a pass/fail matrix.

Use this when a big feature lands: exercise every wired scenario end-to-end
(completes without error + output checks pass) so you catch regressions across
the whole feature surface, not just the one you changed.

Each scenario runs once through the real backend (its own route + payload). A
scenario PASSES if the generation completes AND all its checks pass. NEEDS_WIRING
scenarios are skipped (reported as SKIP) until wired in scenarios.py.

Results viewer: every run collects each scenario's INPUT asset(s) and OUTPUT into
`sanity_runs/<ts>/` and writes an `index.html` there that shows input | output
side by side. Open that file to watch what each scenario produced.

Usage (backend running on the box):
    python sanity.py                     # run all wired scenarios
    python sanity.py --only t2v_540p_20s i2v
    python sanity.py --tags modality     # run scenarios matching a tag
    python sanity.py --tags bump --model-id 2.5 --use-conv-vae --gate-integrity
    python sanity.py --include-unwired    # attempt everything (expect failures)

Emits sanity_runs/<ts>/ (index.html + copied media + results.json) and a table.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import checks
import keep_awake
import perf_config
import queued
import scenarios as scn
from gpu_sampler import GpuSampler

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".flac"}

# --fast: cheapest valid knobs so every path runs in a fraction of the time. This is
# a "does each feature still generate" surface check, NOT a quality/perf signal — it
# deliberately squashes the resolution/duration corners. 540p and 5s are the minimum
# of their respective Literals (see GEN_PAYLOAD_TEMPLATE); extend/retake take float
# seconds, so their duration is capped instead of fixed.
_FAST_VIDEO = {"resolution": "540p", "duration": 5}
_FAST_DURATION_CAP_ROUTES = {"/api/extend", "/api/retake"}
_ICLORA_ROUTE = "/api/ic-lora/generate"


def _fast_overrides(s: scn.Scenario) -> dict:
    """Scenario overrides with the fast caps applied (leaves the original untouched)."""
    ov = dict(s.overrides)
    if "regression" in s.tags:  # already minimal; --fast would inflate 270p/2s to 540p/5s
        return ov
    if s.route is None:                                   # default video route (Literal knobs)
        ov.update(_FAST_VIDEO)
    elif s.route in _FAST_DURATION_CAP_ROUTES and isinstance(ov.get("duration"), (int, float)):
        ov["duration"] = min(float(ov["duration"]), 2.0)  # extend/retake: shrink the added span
    elif s.route == _ICLORA_ROUTE:
        # Half-canvas Stage 1. Only bites on the skip_stage_2 path (transformation
        # LoRAs like day-to-night); a no-op on the canny/depth two-stage path, where
        # Stage 2 sets the output size. See api_types resolution_factor.
        ov["resolution_factor"] = 1.0
    return ov


def _integrity(payload: dict, probe: dict) -> tuple[bool, str]:
    """Compare a default-route video output to the request (duration/fps/audio). ADVISORY
    only — reported, never a gate: a smoke sweep passes as long as a valid file generated.
    Tolerances absorb frame-rounding. Audio: only flags *requested* audio that's MISSING —
    a silent track muxed into a t2v output when audio wasn't requested is expected, so only
    requested-but-absent audio is flagged."""
    issues = []
    want_dur, want_fps = payload.get("duration"), payload.get("fps")
    if want_dur is not None and probe.get("duration") is not None and abs(probe["duration"] - float(want_dur)) > 0.5:
        issues.append(f"duration {probe['duration']:.1f}s!={want_dur}s")
    if want_fps is not None and probe.get("fps") is not None and abs(probe["fps"] - float(want_fps)) > 1.0:
        issues.append(f"fps {probe['fps']}!={want_fps}")
    if bool(payload.get("audio")) and not probe.get("has_audio"):
        issues.append("requested audio missing")
    return (not issues, "ok" if not issues else ", ".join(issues))


def run_one(s: scn.Scenario, fast: bool = False, gate_integrity: bool = False) -> dict:
    t0 = time.time()
    row: dict = {"key": s.key, "title": s.title, "status": "", "wall_s": 0.0,
                 "detail": "", "output": None, "inputs": s.input_paths(),
                 "peak_mib": None, "fit": "", "media": None,
                 "headroom_pct": None, "realtime": None, "energy_wh": None}
    overrides = s.build_overrides(_fast_overrides(s) if fast else s.overrides)
    route = s.route or perf_config.GENERATE_PATH
    payload = overrides if s.route else {**perf_config.GEN_PAYLOAD_TEMPLATE, **overrides}
    sampler = GpuSampler(hz=5).start()
    try:
        if s.queued:  # queued surface: one create body per job (see queued.py)
            for job in s.queued:
                print(f"[sanity]   request  POST {job.route}  (queued)")
                print(f"[sanity]   payload  {json.dumps(queued.request_body(job))}")
        else:
            print(f"[sanity]   request  POST {route}")
            print(f"[sanity]   payload  {json.dumps(payload)}")
        sampler.mark("start")
        if s.cancel_after_s is not None:
            if s.queued:
                resp = queued.trigger_cancellable(s.queued[0], after_s=s.cancel_after_s)
            else:
                resp = perf_config.trigger_generation_cancellable(
                    after_s=s.cancel_after_s, payload=payload, path=route
                )
            sampler.mark("end")
            row["wall_s"] = round(time.time() - t0, 1)
            row["status"] = "PASS"
            row["detail"] = f"cancelled after {s.cancel_after_s}s running ({resp.get('status')})"
            return row
        outs: list[str | None] = []
        if s.queued:
            outs = queued.trigger(s.queued)
            out = outs[-1]
        else:
            out = perf_config.trigger(s.route, overrides) if s.route else perf_config.trigger_generation(overrides)
        sampler.mark("end")
        row["output"] = out
        row["wall_s"] = round(time.time() - t0, 1)
        details, ok_all = [], True
        for chk in s.checks:
            ok, msg = chk(out)
            ok_all = ok_all and ok
            details.append(f"{chk.__name__}={'ok' if ok else 'FAIL'}({msg})")
        for batch_chk in s.batch_checks:
            ok, msg = batch_chk(outs)
            ok_all = ok_all and ok
            details.append(f"{batch_chk.__name__}={'ok' if ok else 'FAIL'}({msg})")
        wall = row["wall_s"]
        # Peak VRAM this scenario needed -> which card tier runs it, and how close to OOM.
        peak = sampler.window_peak("start", "end")
        row["peak_mib"], row["fit"] = (peak or None), perf_config.vram_fit(peak)
        total = perf_config.gpu_total_mib()
        row["headroom_pct"] = round(100.0 * peak / total) if (peak and total) else None
        if peak:
            hd = f", {row['headroom_pct']:.0f}% of {total} MiB" if row["headroom_pct"] else ""
            details.append(f"peak={peak}MiB ({row['fit']}{hd})")
        # Energy this generation drew (mean power x wall).
        power = sampler.window_power_mean("start", "end")
        row["energy_wh"] = round(power * wall / 3600.0, 2) if power else None
        if row["energy_wh"]:
            details.append(f"energy={row['energy_wh']}Wh")
        # Output: resolution, throughput (video seconds per wall second), and an
        # integrity note. Integrity is advisory unless --gate-integrity.
        probe = perf_config.ffprobe_info(out)
        row["media"] = probe
        if probe:
            details.append(f"res={probe['width']}x{probe['height']}")
            if probe.get("duration") and wall:
                row["realtime"] = round(probe["duration"] / wall, 2)
                details.append(f"{row['realtime']}x realtime")
            if s.route is None and not s.queued:  # default video route: duration/fps/audio are in the request
                ok_i, msg_i = _integrity(payload, probe)
                details.append(f"integrity={'ok' if ok_i else 'DIFF'}({msg_i})")
                if gate_integrity and not ok_i:
                    ok_all = False
        row["status"] = "PASS" if ok_all else "FAIL"
        row["detail"] = "; ".join(details)
    except Exception as exc:  # noqa: BLE001
        row["wall_s"] = round(time.time() - t0, 1)
        row["status"] = "ERROR"
        row["detail"] = f"{type(exc).__name__}: {exc}"
    finally:
        sampler.stop()
    return row


def _media_tag(rel: str) -> str:
    ext = Path(rel).suffix.lower()
    r = html.escape(rel)
    if ext in VIDEO_EXT:
        return f'<video controls preload="metadata" width="360" src="{r}"></video>'
    if ext in IMAGE_EXT:
        return f'<img width="360" src="{r}">'
    if ext in AUDIO_EXT:
        return f'<audio controls src="{r}"></audio>'
    return f'<a href="{r}">{r}</a>'


def _backend_up() -> bool:
    try:
        perf_config._get("/health")
    except Exception:  # noqa: BLE001  connection refused, reset, timeout: all mean "not serving"
        return False
    return True


def collect_results(rows: list[dict], ts: str) -> Path:
    """Copy each scenario's input(s) + output into sanity_runs/<ts>/ and write an
    index.html that shows them side by side. Returns the run folder."""
    run_dir = perf_config.RUNS_DIR / f"sanity_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: str | None, dest_name: str) -> str | None:
        if not src or not os.path.exists(src):
            return None
        dest = run_dir / dest_name
        try:
            shutil.copy2(src, dest)
            return dest.name
        except Exception:  # noqa: BLE001
            return None

    cards = []
    for r in rows:
        in_rels = []
        for i, src in enumerate(r.get("inputs") or []):
            rel = _copy(src, f"{r['key']}__input{i}{Path(src).suffix}")
            if rel:
                in_rels.append(rel)
        out_rel = _copy(r.get("output"), f"{r['key']}__output{Path(r['output']).suffix}") if r.get("output") else None
        r["input_files"] = in_rels
        r["output_file"] = out_rel

        color = {"PASS": "#3fb950", "FAIL": "#f85149", "ERROR": "#f85149", "SKIP": "#8b95a5"}.get(r["status"], "#e6e9ef")
        in_html = "".join(f"<div>{_media_tag(x)}<div class=cap>{html.escape(x)}</div></div>" for x in in_rels) or "<div class=mut>— none —</div>"
        out_html = _media_tag(out_rel) if out_rel else f"<div class=mut>{html.escape(str(r.get('output') or 'no output'))}</div>"
        m = r.get("media") or {}
        meta_bits = []
        if r.get("peak_mib"):
            meta_bits.append(f"peak {r['peak_mib']} MiB")
        if r.get("fit"):
            meta_bits.append(r["fit"])
        if r.get("headroom_pct"):
            meta_bits.append(f"{r['headroom_pct']:.0f}% VRAM")
        if m.get("width"):
            meta_bits.append(f"{m['width']}×{m['height']}")
        if m.get("fps"):
            meta_bits.append(f"{m['fps']}fps")
        if m.get("duration"):
            meta_bits.append(f"{m['duration']:.1f}s")
        if r.get("realtime"):
            meta_bits.append(f"{r['realtime']}× realtime")
        if m.get("size"):
            meta_bits.append(f"{m['size'] / 1048576:.1f} MB")
        if m.get("bit_rate"):
            meta_bits.append(f"{m['bit_rate'] / 1e6:.1f} Mbps")
        if r.get("energy_wh"):
            meta_bits.append(f"{r['energy_wh']} Wh")
        if m.get("has_audio"):
            meta_bits.append("audio")
        meta_html = f"<div class=meta>{html.escape(' · '.join(meta_bits))}</div>" if meta_bits else ""
        cards.append(f"""
    <section>
      <h2>{html.escape(r['key'])} <span style="color:{color}">[{r['status']}]</span>
          <span class=mut>{html.escape(r['title'])} · {r['wall_s']}s</span></h2>
      {meta_html}
      <div class=cols>
        <div><h3>input</h3><div class=row>{in_html}</div></div>
        <div><h3>output</h3>{out_html}</div>
      </div>
      <div class=detail>{html.escape(r.get('detail') or '')}</div>
    </section>""")

    doc = f"""<!doctype html><meta charset=utf-8><title>Sanity results {ts}</title>
<style>
 body{{font:14px system-ui,sans-serif;background:#0f1216;color:#e6e9ef;margin:0;padding:16px}}
 h1{{font-size:16px}} h2{{font-size:14px;margin:0 0 8px}} h3{{font-size:12px;color:#8b95a5;margin:0 0 4px}}
 section{{border:1px solid #2a313b;border-radius:8px;padding:12px;margin:12px 0;background:#171b21}}
 .cols{{display:flex;gap:24px;flex-wrap:wrap}} .row{{display:flex;gap:12px;flex-wrap:wrap}}
 .cap{{font-size:11px;color:#8b95a5;max-width:360px;word-break:break-all}}
 .meta{{font-size:12px;color:#8b95a5;margin:0 0 10px}}
 .mut{{color:#8b95a5}} .detail{{font-size:12px;color:#8b95a5;margin-top:8px;word-break:break-word}}
 video,img{{border:1px solid #2a313b;border-radius:6px;background:#000}}
</style>
<h1>Sanity results — {ts}</h1>
{''.join(cards)}
"""
    (run_dir / "index.html").write_text(doc, encoding="utf-8")
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return run_dir


def main() -> int:
    """Returns a process exit code: 0 = all ran scenarios PASS, 1 = any FAIL/ERROR."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="specific scenario keys")
    ap.add_argument("--tags", nargs="*", help="run scenarios matching any tag")
    ap.add_argument("--include-unwired", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="cheapest valid knobs (540p/5s) — quick 'does each path run' surface check")
    ap.add_argument("--gate-integrity", action="store_true",
                    help="fail a default-route scenario whose output duration/fps/audio "
                         "mismatches the request (advisory unless this flag is set)")
    ap.add_argument("--model-id",
                    help="activate an LTX version first (Settings label or model_id, e.g. 2.5); "
                         "must already be installed")
    ap.add_argument("--use-conv-vae", action=argparse.BooleanOptionalAction, default=None,
                    help="Fast decode on/off (same as Settings); 2.5 only, weights must be on disk")
    args = ap.parse_args()

    # A Windows console is often cp1252: never let a "≥" in a detail string kill the sweep.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")

    perf_config.wait_for_backend()
    try:
        if args.model_id:
            perf_config.activate_ltx_model(args.model_id)
        if args.use_conv_vae is not None:
            perf_config.set_use_conv_vae(args.use_conv_vae)
    except RuntimeError as exc:
        print(f"[sanity] {exc}", file=sys.stderr)
        return 1

    if args.only:
        unknown = [k for k in args.only if k not in scn.SCENARIOS]
        if unknown:
            print(f"[sanity] unknown scenario key(s): {', '.join(unknown)}", file=sys.stderr)
            return 1
        pool = [scn.SCENARIOS[k] for k in args.only]
    elif args.tags:
        pool = [s for s in scn.all_scenarios() if set(s.tags) & set(args.tags)]
    elif args.include_unwired:
        pool = scn.all_scenarios()
    else:
        pool = scn.ready()

    if not pool:  # an empty selection must never look like a green run
        print("[sanity] no scenarios selected (check --only / --tags)", file=sys.stderr)
        return 1
    missing_tools = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing_tools and any("regression" in s.tags for s in pool):
        # Every oracle decodes the output with these; without them the run can only report noise.
        print(f"[sanity] {' and '.join(missing_tools)} not on PATH: the regression checks need them. "
              "Install ffmpeg (Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`), "
              "open a new terminal and re-run.", file=sys.stderr)
        return 1
    if not args.only:
        pool.sort(key=lambda s: s.order)  # stable: registration order within the same position
    checks.reset()
    queued.reset()
    keep_awake.start()  # a long sweep must not idle into sleep and kill the backend

    ts = time.strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []
    caps = perf_config.scenario_caps()
    for pos, s in enumerate(pool):
        print(f"[sanity] Running test {pos + 1}/{len(pool)}", flush=True)
        if s.needs_wiring and not args.include_unwired and not args.only:
            rows.append({"key": s.key, "title": s.title, "status": "SKIP",
                         "wall_s": 0.0, "detail": "needs_wiring", "output": None, "inputs": s.input_paths()})
            print(f"[sanity] SKIP  {s.key:<18} (needs wiring)")
            continue
        cap_reason = s.blocked_by_caps(caps)
        if cap_reason:
            rows.append({"key": s.key, "title": s.title, "status": "SKIP", "wall_s": 0.0,
                         "detail": cap_reason, "output": None, "inputs": s.input_paths()})
            print(f"[sanity] SKIP  {s.key:<18} ({cap_reason})")
            continue
        missing = s.unavailable()
        if missing:
            detail = "needs download (open the app → LoRA / IC-LoRA library): " + ", ".join(missing)
            rows.append({"key": s.key, "title": s.title, "status": "SKIP", "wall_s": 0.0,
                         "detail": detail, "output": None, "inputs": s.input_paths()})
            print(f"[sanity] SKIP  {s.key:<18} ({detail})")
            continue
        print(f"[sanity] run   {s.key:<18} ...", flush=True)
        row = run_one(s, fast=args.fast, gate_integrity=args.gate_integrity)
        rows.append(row)
        print(f"[sanity] {row['status']:<5} {s.key:<18} {row['wall_s']:>6.1f}s  out={row['output']}  {row['detail']}")
        if row["status"] == "ERROR" and not _backend_up():
            # A crashed backend (OOM, kill) would otherwise turn every remaining scenario into noise.
            print("[sanity] backend unreachable: aborting the sweep", file=sys.stderr)
            for rest in pool[pos + 1:]:
                rows.append({"key": rest.key, "title": rest.title, "status": "SKIP", "wall_s": 0.0,
                             "detail": "backend unreachable (crashed earlier in the sweep)",
                             "output": None, "inputs": rest.input_paths()})
            break

    run_dir = collect_results(rows, ts)

    npass = sum(r["status"] == "PASS" for r in rows)
    nfail = sum(r["status"] in ("FAIL", "ERROR") for r in rows)
    nskip = sum(r["status"] == "SKIP" for r in rows)
    print("\n===== SANITY SUMMARY =====")
    print(f"pass={npass}  fail/err={nfail}  skip={nskip}")
    peaks = [r["peak_mib"] for r in rows if r.get("peak_mib")]
    if peaks:
        worst = max(peaks)
        print(f"peak VRAM (max over scenarios): {worst} MiB -> {perf_config.vram_fit(worst)}")
    rts = [r["realtime"] for r in rows if r.get("realtime")]
    if rts:
        print(f"throughput (median): {statistics.median(rts):.2f}x realtime")
    print(f"results + input/output videos: {run_dir}")
    print(f"open: {run_dir / 'index.html'}")
    print("VERDICT:", "PASS" if nfail == 0 else f"FAIL ({nfail} scenario(s))")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
