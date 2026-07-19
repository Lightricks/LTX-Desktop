#!/usr/bin/env python3
"""Perf dashboard control server (localhost-only).

A thin FastAPI app that lets the HTML dashboard fully drive the perf suite — set
the backend auth token, flip the cache + Torch Compile toggles, launch soak /
A/B / decompose / sanity runs, poll their logs/status, stop them, view sanity
results, and chart the soak trend. It reuses perf_config.py directly, so it
inherits the same single-seam wiring.

All run artifacts (logs, soak CSVs, sanity results) live in one folder,
`perf_config.RUNS_DIR`; run metadata is persisted alongside so the runs table
survives a server restart.

SECURITY: binds 127.0.0.1 only. It spawns processes and flips app settings, so
never expose it on a network interface — reach it from a browser on the same box.

Run: python dashboard/server.py  (or the run_dashboard launchers).
"""

from __future__ import annotations

import csv as csv_mod
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
PERF_DIR = HERE.parent                                     # .../perf
sys.path.insert(0, str(PERF_DIR))
import catalog  # noqa: E402
import perf_config  # noqa: E402
import scenarios as scn  # noqa: E402

RUNS_DIR = perf_config.RUNS_DIR                            # logs + CSVs + sanity_<ts>/
PY = sys.executable


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _load_persisted_runs()                    # rebuild the runs table from disk on start
    yield
    for r in RUNS.values():                   # terminate still-running spawned runs on exit
        _terminate(r)


app = FastAPI(title="LTX Desktop Perf Dashboard", lifespan=_lifespan)
# Serve run artifacts (sanity input/output media + index.html) so they open
# in-browser over http -> /results/sanity_<ts>/  (file:// media tags are blocked).
app.mount("/results", StaticFiles(directory=str(RUNS_DIR), html=True), name="results")

# id -> {id, kind, label, argv, log, started, proc}. proc is None for runs
# reconstructed from disk after a restart.
RUNS: dict[str, dict] = {}


class RunReq(BaseModel):
    kind: str                      # soak | ab | decompose | sanity
    params: dict = {}


class ToggleReq(BaseModel):
    enabled: bool


class TokenReq(BaseModel):
    token: str


# --------------------------------------------------------------------------- #
# Run lifecycle
# --------------------------------------------------------------------------- #
def _meta_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _child_env() -> dict[str, str]:
    """Env for spawned perf scripts — carries the auth token so they authenticate,
    and forces unbuffered stdout so per-gen log lines reach the file live."""
    env = dict(os.environ)
    if perf_config.AUTH_TOKEN:
        env["PERF_AUTH_TOKEN"] = perf_config.AUTH_TOKEN
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _spawn(kind: str, argv: list[str], label: str) -> str:
    run_id = f"{kind}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
    log_path = RUNS_DIR / f"{run_id}.log"
    logf = open(log_path, "w", buffering=1)  # line-buffered
    proc = subprocess.Popen(
        [PY, "-u", *argv], cwd=str(PERF_DIR), stdout=logf, stderr=subprocess.STDOUT,
        text=True, env=_child_env(),
    )
    meta = {"id": run_id, "kind": kind, "label": label, "argv": argv,
            "log": str(log_path), "started": time.time()}
    _meta_path(run_id).write_text(json.dumps(meta))
    RUNS[run_id] = {**meta, "proc": proc, "logf": logf}
    return run_id


def _load_persisted_runs() -> None:
    """Rebuild the runs table from metadata sidecars so a restart keeps history."""
    for meta_file in RUNS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:  # noqa: BLE001
            continue
        rid = meta.get("id")
        if rid and rid not in RUNS:
            RUNS[rid] = {**meta, "proc": None}


def _read_log(r: dict) -> str:
    """Read a run's log once. Callers pass the text into the parsers below so a single
    /api/runs poll reads each log once, not once per parsed field."""
    try:
        return Path(r["log"]).read_text(errors="replace")
    except OSError:
        return ""


def _reap(r: dict) -> None:
    """Close the run's log file handle once its process has exited — otherwise the
    handle opened in _spawn leaks for the life of the dashboard."""
    proc, logf = r.get("proc"), r.get("logf")
    if logf is not None and not logf.closed and (proc is None or proc.poll() is not None):
        logf.close()
        r["logf"] = None


def _status(r: dict, text: str) -> str:
    proc = r.get("proc")
    if proc is not None:
        code = proc.poll()
        if code is None:
            return "running"
        return "done" if code == 0 else f"failed({code})"
    # Reconstructed after restart — infer completion from the log's terminal marker.
    if not text:
        return "ended"
    return "done" if ("VERDICT" in text or "SUMMARY" in text) else "ended"


def _terminate(r: dict) -> None:
    proc = r.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    _reap(r)


_SOAK_GEN = re.compile(r"\]\s*gen\s+\d+")          # "[soak] gen   3  wall=..."
_DECOMP_GEN = re.compile(r"gen\s+\d+:")            # "  measure gen 2: 41.0s"
_SANITY_DONE = re.compile(r"\b(PASS|FAIL|SKIP)\b")


def _argv_val(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _verdict(text: str) -> str:
    """The run's final verdict, parsed from the log's VERDICT line (soak/AB/sanity)."""
    for line in reversed(text.splitlines()):
        if "VERDICT" in line:
            return line.split("VERDICT", 1)[1].lstrip(": ").strip()
    return ""


def _result_url(text: str) -> str:
    """Browser URL of a run's result viewer, or "" if none yet. Parsed from the log
    (sanity prints `open: <dir>/index.html` at the end) so it appears exactly when the
    result lands, and maps this run to its folder without renaming anything."""
    for line in reversed(text.splitlines()):
        if line.startswith("open:") and "index.html" in line:
            folder = Path(line.split("open:", 1)[1].strip()).parent.name
            if (RUNS_DIR / folder / "index.html").exists():
                return f"/results/{folder}/"
    return ""


def _progress(r: dict, text: str) -> str:
    """Live progress parsed from the run's log (the CSV only lands at the end)."""
    kind = r["kind"]
    if kind == "soak":
        done = len(_SOAK_GEN.findall(text))
        n = _argv_val(r.get("argv", []), "--n")
        if not n:
            return str(done)
        total = int(n) * 2 if "--pair" in r.get("argv", []) else int(n)  # pair runs on + off
        return f"{done}/{total}"
    if kind == "decompose":
        return f"{len(_DECOMP_GEN.findall(text))} gens"
    if kind == "coldstart":
        return f"{len(_SOAK_GEN.findall(text))} gens"  # "[coldstart] gen N (...)"
    if kind == "sanity":
        n = len(_SANITY_DONE.findall(text))
        return f"{n} done" if n else ""
    if kind == "ab":
        return "done" if ("SUMMARY" in text or "PSNR" in text.upper()) else "running"
    return ""


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
# Dev dashboard: never let the browser cache the UI files, so a server update always
# reaches the page without a manual hard-refresh.
_NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(HERE / "index.html"), headers=_NO_CACHE)


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(str(HERE / "app.js"), media_type="application/javascript", headers=_NO_CACHE)


@app.get("/style.css")
def style_css() -> FileResponse:
    return FileResponse(str(HERE / "style.css"), media_type="text/css", headers=_NO_CACHE)


@app.get("/chart.umd.min.js")
def chart_js() -> FileResponse:
    # Vendored Chart.js (version-pinned, immutable) so the dashboard renders the soak
    # trend offline / on an air-gapped box — no CDN dependency in the measurement tool.
    return FileResponse(str(HERE / "chart.umd.min.js"), media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/health")
def health() -> JSONResponse:
    token_set = bool(perf_config.AUTH_TOKEN)
    try:
        detail = perf_config._get(perf_config.HEALTH_PATH)  # noqa: SLF001
        return JSONResponse({"backend": "ok", "detail": detail, "token_set": token_set})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"backend": "down", "detail": str(exc), "token_set": token_set})


_ENV_CACHE: dict | None = None


@app.get("/api/env")
def env_info() -> dict:
    """Platform capabilities from the LTX backend's /api/gpu-info, so the UI can hide
    CUDA-only controls: Torch Compile and the diffusion-stage-cache patch have no effect
    off CUDA (e.g. Apple Silicon / MPS), so their toggles and the patch/compile-specific
    runs aren't relevant there — only the scenario sweeps are."""
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    try:
        gi = perf_config._get("/api/gpu-info") or {}  # noqa: SLF001
        _ENV_CACHE = {"cuda": bool(gi.get("cuda_available")), "mps": bool(gi.get("mps_available"))}
        return _ENV_CACHE
    except Exception:  # noqa: BLE001
        # gpu-info unreachable (token not set yet, backend busy). Fall back to the OS:
        # macOS is never CUDA (MPS or CPU), so we can hide the CUDA-only controls without
        # the backend. On Windows/Linux we can't tell CPU from CUDA here, so stay unknown.
        if sys.platform == "darwin":
            _ENV_CACHE = {"cuda": False, "mps": True}
            return _ENV_CACHE
        return {"cuda": None, "mps": None}            # unknown — don't cache, let the UI retry


@app.get("/api/gpu")
def gpu() -> dict:
    try:
        out: dict = dict(perf_config.gpu_stats())   # CUDA: full nvidia-smi telemetry + host RAM
    except Exception:  # noqa: BLE001
        # No nvidia-smi (e.g. Apple Silicon / MPS): still surface host RAM — including the
        # *available* RAM that gates local generation — plus the GPU name and MPS memory
        # (how close the process is to the MPS ceiling) from the backend.
        out = dict(perf_config.host_ram())
        try:
            gi = perf_config._get("/api/gpu-info") or {}  # noqa: SLF001
            out["name"] = gi.get("gpu_name")
            out["mps"] = bool(gi.get("mps_available"))
        except Exception:  # noqa: BLE001
            pass
        try:
            mem = perf_config._get("/api/gpu-info/mps") or {}  # noqa: SLF001
            if mem.get("available"):
                out["mps_driver_mib"] = mem.get("driver_mib")
                out["mps_max_mib"] = mem.get("recommended_max_mib")
        except Exception:  # noqa: BLE001
            pass
    out.update(perf_config.disk_free())              # disk headroom on the models/outputs volume
    return out


@app.get("/api/gpu-busy")
def gpu_busy() -> dict:
    """Brief utilization sample — used to warn before a soak if the GPU is already
    in use by another process (which would contaminate the wall-time signal)."""
    try:
        busy, util = perf_config.gpu_busy()
        return {"busy": busy, "util": util}
    except Exception as exc:  # noqa: BLE001
        return {"busy": False, "util": None, "error": str(exc)}


@app.post("/api/token")
def set_token(req: TokenReq) -> dict:
    """Set the backend Bearer token at runtime (used by the server + spawned runs)."""
    perf_config.AUTH_TOKEN = req.token.strip()
    return {"token_set": bool(perf_config.AUTH_TOKEN)}


@app.get("/api/toggles")
def toggles() -> dict:
    """Live cache + Torch-Compile state read from the backend, so the UI reflects its
    actual settings."""
    try:
        s = perf_config.get_settings()
    except Exception as exc:  # noqa: BLE001
        return {"cache": None, "torch_compile": None, "error": str(exc)}

    def pick(*keys: str) -> bool | None:
        for k in keys:
            if k in s:
                return bool(s[k])
        return None

    return {
        "cache": pick("diffusionStageCacheEnabled", "diffusion_stage_cache_enabled"),
        "torch_compile": pick("useTorchCompile", "use_torch_compile"),
    }


@app.post("/api/cache")
def set_cache(req: ToggleReq) -> dict:
    perf_config.set_cache_enabled(req.enabled)
    return {"cache_enabled": req.enabled}


@app.post("/api/torch-compile")
def set_torch_compile(req: ToggleReq) -> dict:
    perf_config.set_torch_compile_enabled(req.enabled)
    return {"torch_compile_enabled": req.enabled}


@app.get("/api/scenarios")
def list_scenarios() -> list[dict]:
    """Scenarios plus availability, crossed with the app's LoRA/IC-LoRA library (SSOT).
    `available` is False when a required catalog weight isn't downloaded; the reason points
    the user at the app's library. Availability uses the catalog cache — POST /api/catalog/
    refresh after downloading a weight in the app to re-check without restarting."""
    out: list[dict] = []
    for s in scn.all_scenarios():
        missing = s.unavailable()
        reason = ("needs download (open the app → LoRA / IC-LoRA library): " + ", ".join(missing)
                  if missing else "")
        out.append({"key": s.key, "title": s.title, "needs_wiring": s.needs_wiring,
                    "tags": s.tags, "available": not missing, "unavailable_reason": reason})
    return out


@app.post("/api/catalog/refresh")
def refresh_catalog() -> dict:
    """Drop the cached LoRA/IC-LoRA listing so the next /api/scenarios re-reads downloaded
    status — used by the dashboard's refresh button after a weight is downloaded in the app."""
    catalog.refresh()
    return {"refreshed": True}


@app.post("/api/run")
def run(req: RunReq) -> dict:
    p = req.params
    if req.kind == "soak":
        if p.get("pair"):
            argv = ["soak_test.py", "--n", str(p.get("n", 50)), "--pair"]
            label = f"soak PAIR n={p.get('n', 50)} (cache_on + cache_off)"
        else:
            argv = ["soak_test.py", "--n", str(p.get("n", 50)),
                    "--label", p.get("label", "cache_on"),
                    "--set-cache", p.get("set_cache", "on")]
            label = f"soak n={p.get('n', 50)} cache={p.get('set_cache', 'on')}"
    elif req.kind == "ab":
        argv = ["output_ab.py", "--seed", str(p.get("seed", 42)), "--run"]
        label = f"A/B seed={p.get('seed', 42)}"
    elif req.kind == "decompose":
        argv = ["decompose.py", "--reps", str(p.get("reps", 5))]
        label = f"decompose reps={p.get('reps', 5)}"
    elif req.kind == "coldstart":
        argv = ["coldstart.py", "--warm", str(p.get("warm", 3))]
        label = f"cold-start warm={p.get('warm', 3)}"
    elif req.kind == "sanity":
        argv = ["sanity.py"]
        if p.get("only"):
            argv += ["--only", *p["only"]]
        if p.get("tags"):
            argv += ["--tags", *p["tags"]]
        if p.get("include_unwired"):
            argv += ["--include-unwired"]
        if p.get("fast"):
            argv += ["--fast"]
        fast = " (fast)" if p.get("fast") else ""
        if p.get("only"):
            label = "scenario: " + ",".join(p["only"]) + fast
        elif p.get("tags"):
            label = "sanity tags: " + ",".join(p["tags"]) + fast
        else:
            label = "sanity sweep" + (" (all)" if p.get("include_unwired") else "") + fast
    else:
        return {"error": f"unknown kind {req.kind}"}
    return {"id": _spawn(req.kind, argv, label)}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    r = RUNS.get(run_id)
    if not r:
        return {"error": "no such run"}
    _terminate(r)
    return {"id": run_id, "status": _status(r, _read_log(r))}


@app.post("/api/runs/stop-all")
def stop_all() -> dict:
    stopped = []
    for r in RUNS.values():
        proc = r.get("proc")
        if proc is not None and proc.poll() is None:
            _terminate(r)
            stopped.append(r["id"])
    return {"stopped": stopped}


def _elapsed(r: dict, now: float) -> float:
    """Seconds the run took. Frozen once the run finishes so the table stops
    counting: snapshot the end time on first observation (in-process runs), or
    use the log's last-modified time (runs reconstructed after a restart)."""
    proc = r.get("proc")
    if proc is not None:
        if proc.poll() is None:
            return now - r["started"]                    # still running
        r.setdefault("ended", now)                       # freeze at completion
        return r["ended"] - r["started"]
    try:
        return Path(r["log"]).stat().st_mtime - r["started"]
    except OSError:
        return 0.0


@app.get("/api/runs")
def runs() -> list[dict]:
    now = time.time()
    out: list[dict] = []
    for r in sorted(RUNS.values(), key=lambda x: x["started"], reverse=True):
        text = _read_log(r)                       # read the log once, not once per field
        _reap(r)                                  # close the fd if the run has finished
        out.append({
            "id": r["id"], "kind": r["kind"], "label": r["label"],
            "status": _status(r, text), "started": r["started"],
            "elapsed_s": round(max(_elapsed(r, now), 0.0), 1), "progress": _progress(r, text),
            "verdict": _verdict(text), "result_url": _result_url(text),
        })
    return out


@app.get("/api/backend-log")
def backend_log(tail: int = 500) -> dict:
    """Tail the newest backend session log (session_*.log / backend_*.log) so a
    failed run's backend traceback is viewable in the dashboard."""
    d = perf_config.BACKEND_LOGS_DIR
    try:
        logs = [f for f in d.glob("*.log") if f.name.startswith(("session_", "backend_"))]
    except OSError:
        logs = []
    if not logs:
        return {"file": None, "log": f"(no backend logs in {d})"}
    newest = max(logs, key=lambda f: f.stat().st_mtime)
    try:
        lines = newest.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    return {"file": newest.name, "log": "\n".join(lines[-tail:])}


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, tail: int = 400) -> dict:
    r = RUNS.get(run_id)
    if not r:
        return {"error": "no such run"}
    text = _read_log(r)
    lines = text.splitlines()
    return {"status": _status(r, text), "log": "\n".join(lines[-tail:])}


def _read_soak_csv(path: Path) -> dict:
    """Parse a soak CSV into column arrays for charting (skips a truncated final row
    from a killed run)."""
    gen, floor, wall, ram = [], [], [], []
    with open(path) as f:
        for row in csv_mod.DictReader(f):
            try:
                g, fl, wl = int(row["gen"]), int(row["floor_mib"]), float(row["wall_s"])
            except (TypeError, ValueError, KeyError):
                continue
            gen.append(g)
            floor.append(fl)
            wall.append(wl)
            rv = row.get("ram_floor_mib", "")
            ram.append(float(rv) if rv not in (None, "") else None)
    return {"gen": gen, "floor": floor, "wall": wall, "ram": ram}


def _soak_csv_series(r: dict) -> list[tuple[str, Path]]:
    """Every CSV THIS run wrote, as (label, path). A single soak yields one; a --pair
    yields cache_on + cache_off — parsed from the log's `[soak] <label>: csv -> <path>`
    lines so clicking the row charts this run (and both halves of a pair). Falls back to
    the newest CSV matching the run's --label if the log lines aren't present yet."""
    try:
        text = Path(r["log"]).read_text(errors="replace")
    except OSError:
        text = ""
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = re.search(r"\[soak\]\s+(\S+):\s+csv -> (.+\.csv)", line)
        if m:
            p = RUNS_DIR / Path(m.group(2).strip()).name
            if p.exists() and p.name not in seen:
                seen.add(p.name)
                out.append((m.group(1), p))
    if out:
        return out
    label = _argv_val(r.get("argv", []), "--label") or "*"
    matches = sorted(RUNS_DIR.glob(f"soak_{label}_*.csv"), key=os.path.getmtime, reverse=True)
    return [(_argv_val(r.get("argv", []), "--label") or "soak", matches[0])] if matches else []


@app.get("/api/runs/{run_id}/soakcsv")
def soak_csv(run_id: str) -> dict:
    """Chartable soak series for THIS run — one per CSV it wrote (two for a --pair, so
    the cache_on and cache_off floors overlay for comparison)."""
    r = RUNS.get(run_id)
    if not r or r.get("kind") != "soak":
        return {"series": []}
    series = [{"label": label, **_read_soak_csv(path)}
              for label, path in _soak_csv_series(r) if path.exists()]
    return {"series": series}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PERF_DASH_PORT", "8750"))
    print(f"Perf dashboard on http://127.0.0.1:{port}  (localhost only)")
    uvicorn.run(app, host="127.0.0.1", port=port)
