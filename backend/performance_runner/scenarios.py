"""Scenario registry — one entry per Desktop feature.

This is the "cover all functionalities" seam. A *scenario* is a named generation
config plus optional post-run checks: for the default video route it's payload
overrides merged onto ``perf_config.GEN_PAYLOAD_TEMPLATE``; for other routes
(``route=...``) the overrides ARE the full request body. Adding coverage for a
new feature = add one entry here; the sanity sweep and the dashboard pick it up.

Scenarios are wired end-to-end. If one is ever added as a scaffold
(``needs_wiring=True``) because its endpoint isn't taught to the harness yet, it's
skipped by the default sweep and only attempted with ``--only`` / ``--include-unwired``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from queued import QueuedJob

# Bundled fixtures (backend/perf/test_assets) so the modality scenarios are
# runnable without hand-supplying media. Absolute paths so they resolve regardless
# of backend cwd.
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assets")
IMAGE_ASSET = os.path.join(_ASSETS, "reference_image.png")   # fox & panda cartoon, 540p (i2v/ia2v)
AUDIO_ASSET = os.path.join(_ASSETS, "reference_audio.wav")   # ~5s speech, STEREO 44.1kHz (a2v needs 2ch)
VIDEO_ASSET = os.path.join(_ASSETS, "reference_video.mp4")   # 540p/5s cozy felt duck, day setting (video-input)

# Non-catalog weights the canny/depth control path needs, RELATIVE to the backend's
# models_dir. These are NOT in lora_catalog.json (they're shared control checkpoints, not
# selectable catalog items), so their presence is checked on disk rather than via the
# catalog listing. Catalog LoRAs / IC-LoRAs are referenced by *id* instead (see below) and
# resolved — filename, ref, and downloaded status — from the app's /api/loras + /api/ic-loras
# (SSOT). The sweep SKIPS a scenario whose required weights aren't downloaded.
CP_UNION_CONTROL = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"   # canny + depth
CP_DEPTH_PROCESSOR = "dpt-hybrid-midas"                                     # depth only (folder)


@dataclass
class Scenario:
    key: str
    title: str
    overrides: dict[str, Any]
    # post-run checks: fn(output_path|None) -> (ok: bool, detail: str)
    checks: list[Callable[[str | None], tuple[bool, str]]] = field(default_factory=list)
    needs_wiring: bool = False
    tags: list[str] = field(default_factory=list)
    # POST target. None -> the default video route (/api/generate) with the payload
    # merged onto GEN_PAYLOAD_TEMPLATE. When set, `overrides` IS the full request body
    # posted to `route` (generate-image, extend, retake, ic-lora/generate).
    route: str | None = None
    # Catalog ids this scenario needs (SSOT: resolved via /api/loras & /api/ic-loras).
    # For a plain-LoRA t2v scenario, required_loras ARE the LoRAs applied — their `ref` is
    # built from the catalog at run time (see build_overrides). The sweep skips a scenario
    # whose catalog weights aren't downloaded, pointing the user at the app's library.
    required_loras: list[str] = field(default_factory=list)
    required_ic_loras: list[str] = field(default_factory=list)
    # Non-catalog weights (relative to models_dir) checked on disk (union-control, depth).
    required_files: list[str] = field(default_factory=list)
    # If set, sanity starts /api/generate then POSTs /api/generate/cancel after this
    # many seconds of phase=inference (not merely status=running). PASS = cancelled.
    cancel_after_s: float | None = None
    # Run position within a sweep (stable sort, default 0). Lets a reference scenario run
    # first and a "still reproduces after everything else" scenario run last, regardless
    # of registration order. See regression_scenarios.py.
    order: int = 0
    # Queued surface (Home/Explore, /api/generations/...): when set, the scenario is driven
    # through queued.py instead of `route`/`overrides`. Several jobs are submitted together;
    # `checks` run on the last output and `batch_checks` on all outputs (submission order).
    queued: list[QueuedJob] | None = None
    batch_checks: list[Callable[[list[str | None]], tuple[bool, str]]] = field(default_factory=list)
    # Local-offering flags this scenario needs (retake / extend / builtin_control).
    # Greyed out when the active LTX family does not offer them.
    requires_caps: list[str] = field(default_factory=list)

    def input_paths(self) -> list[str]:
        """On-disk input assets referenced by this scenario (for the results viewer)."""
        keys = ("imagePath", "audioPath", "video_path", "input_path", "source_video", "control_video_path")
        out = []
        for k in keys:
            v = self.overrides.get(k)
            if isinstance(v, str) and os.path.exists(v):
                out.append(v)
        for kf in self.overrides.get("keyframes") or []:
            if isinstance(kf, dict):
                p = kf.get("imagePath")
                if isinstance(p, str) and os.path.exists(p):
                    out.append(p)
        for job in self.queued or []:
            out += [p for p in job.inputs.values() if os.path.exists(p) and p not in out]
        return out

    def build_overrides(self, base: dict[str, Any]) -> dict[str, Any]:
        """Resolve required_loras (catalog ids) into the default-route `loras` payload
        (``[{"ref": "loras/<id>/<file>", "scale": ...}]``). No-op for non-default routes or
        scenarios without plain LoRAs. Import is local to avoid a cycle at module load."""
        if not self.required_loras or self.route is not None:
            return base
        import catalog
        refs = [{"ref": catalog.lora_ref(cid), "scale": 1.0} for cid in self.required_loras]
        refs = [r for r in refs if r["ref"]]
        return {**base, "loras": refs} if refs else base

    def unavailable(self) -> list[str]:
        """Display names of required weights that are NOT present (empty = ready to run).

        Catalog LoRAs / IC-LoRAs resolve via the app's library listing (SSOT: /api/loras +
        /api/ic-loras report `downloaded`); non-catalog control weights (union-control, depth
        processor) are checked on disk. Backend/models_dir unreachable -> [] (let it run and
        surface the real error). Single source for both the CLI sweep and the dashboard."""
        import catalog
        import perf_config
        missing = list(catalog.unavailable(self.required_loras, self.required_ic_loras))
        if self.required_files:
            base = perf_config.models_dir()
            if base is not None:
                missing += [rel for rel in self.required_files if not (base / rel).exists()]
        return missing

    def blocked_by_caps(self, caps: dict[str, object] | None) -> str | None:
        """Why this scenario cannot run on the active LTX family, or None if ok.

        ``caps`` is ``perf_config.scenario_caps`` (models-specs + IC-LoRA recommendation).
        Unreachable backend -> None so the sweep still attempts and surfaces the real error.
        """
        if not caps or not self.requires_caps:
            return None
        family = caps.get("family") or "this model"
        for feat in self.requires_caps:
            if not caps.get(feat, True):
                return f"not supported on LTX {family} ({feat})"
        return None


# ---- reusable checks -------------------------------------------------------- #
def output_exists(path: str | None) -> tuple[bool, str]:
    if not path:
        return False, "no output path returned"
    ok = os.path.exists(path)
    return ok, f"{'found' if ok else 'MISSING'}: {path}"


def output_nonempty(path: str | None) -> tuple[bool, str]:
    if not path or not os.path.exists(path):
        return False, "no file"
    sz = os.path.getsize(path)
    return sz > 1024, f"{sz} bytes"


# E2 signature: a full-width eager K=11 NA tile on MPS silently zeros the last
# several frames. Per-frame mean luma below this (0-255) counts as black.
_BLACK_LUMA_MAX = 8.0


def last_frames_not_black(path: str | None, n_frames: int = 8) -> tuple[bool, str]:
    """Fail if any of the last ~n_frames have near-zero luma.

    ``-sseof -1`` seeks one second before EOF; ``-vf reverse,scale=1:1`` then
    ``-frames:v n`` takes the true last n frames as 1-byte gray samples (without
    reverse, ``-frames:v`` would decode the *first* n frames after the seek).
    Missing ffmpeg is a failure — a gate that skips is a false pass.
    """
    if not path or not os.path.exists(path):
        return False, "no file"
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-sseof", "-1", "-i", path,
                "-an", "-vf", "reverse,scale=1:1,format=gray", "-frames:v", str(n_frames),
                "-f", "rawvideo", "-pix_fmt", "gray",
                "pipe:1",
            ],
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError:
        return False, "ffmpeg not on PATH"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out decoding tail frames"
    raw = proc.stdout
    if proc.returncode != 0 or not raw:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:200]
        return False, f"ffmpeg failed: {err or 'no pixels'}"
    # One byte per frame after scale=1:1. Mean-of-all-tail-pixels would pass a
    # mostly-bright clip whose last few frames are black.
    per_frame = list(raw)
    worst = min(per_frame)
    ok = worst > _BLACK_LUMA_MAX
    return ok, f"tail min luma={worst:.1f} / {len(per_frame)} frames ({'ok' if ok else 'BLACK'})"


DEFAULT_CHECKS = [output_exists, output_nonempty]
BUMP_VIDEO_CHECKS = [output_exists, output_nonempty, last_frames_not_black]


# ---- feature scenarios (seeded from Desktop's surface) ---------------------- #
# Resolution/duration corners on the fast/distilled two-stage path (already
# validated manually — good smoke baseline).
# WIRED to GenerateVideoRequest (api_types.py): resolution/duration/fps are
# Literals; the backend derives pixel dims from resolution + aspectRatio, so we
# never pass width/height/num_frames.
_RES = [
    ("t2v_540p_20s", "T2V 540p / 20s (16:9)", {"resolution": "540p", "duration": 20, "fps": 24}),
    ("t2v_720p_10s", "T2V 720p / 10s (16:9)", {"resolution": "720p", "duration": 10, "fps": 24}),
    ("t2v_1080p_5s", "T2V 1080p / 5s (16:9)", {"resolution": "1080p", "duration": 5, "fps": 24}),
    ("t2v_720p_9x16", "T2V 720p / 5s (9:16 vertical)",
     {"resolution": "720p", "duration": 5, "fps": 24, "aspectRatio": "9:16"}),
]

SCENARIOS: dict[str, Scenario] = {}


def _add(s: Scenario) -> None:
    SCENARIOS[s.key] = s


for key, title, ov in _RES:
    _add(Scenario(key, title, {"prompt": "a calm ocean at sunset", "seed": 42, **ov},
                  checks=DEFAULT_CHECKS, tags=["t2v", "resolution"]))

# Modalities — same /api/generate route, using the bundled test_assets fixtures.
# Ready to run (needs_wiring=False): imagePath/audioPath point at real files.
_add(Scenario("i2v", "Image-to-Video",
              {"prompt": "gentle zoom", "seed": 42, "imagePath": IMAGE_ASSET},
              checks=DEFAULT_CHECKS, tags=["i2v", "modality"]))
_add(Scenario("a2v", "Audio-to-Video (dad-joke speech, ~5s)",
              {"prompt": "a cartoon character telling a joke, talking to camera", "seed": 42,
               "audio": True, "audioPath": AUDIO_ASSET, "duration": 6},
              checks=DEFAULT_CHECKS, tags=["a2v", "modality", "speech"]))
_add(Scenario("ia2v", "Image+Audio-to-Video (fox & panda + speech, ~5s)",
              {"prompt": "two cartoon animals chatting at a cafe table, one telling a joke", "seed": 42,
               "audio": True, "imagePath": IMAGE_ASSET, "audioPath": AUDIO_ASSET, "duration": 6},
              checks=DEFAULT_CHECKS, tags=["ia2v", "modality", "speech"]))

# Plain (style) LoRAs on the default t2v route. Reference catalog ids in required_loras;
# the loras=[{ref, scale}] payload is built from the catalog at run time (build_overrides)
# and the scenario is skipped unless the id is downloaded (checked via the app's library).
# One single LoRA and one two-LoRA stack, to exercise the single- and multi-adapter paths.
_add(Scenario("lora_cozy_felt", "LoRA: Cozy Felt style (t2v)",
              {"prompt": "a cozy felt-style red fox trotting through a sunlit autumn forest, "
                         "handmade wool texture, soft warm lighting, shallow depth of field",
               "seed": 42},
              checks=DEFAULT_CHECKS, tags=["lora", "t2v"],
              required_loras=["cozy-felt-style"]))
_add(Scenario("lora_cozy_felt_openwheel", "LoRA combo: Cozy Felt + Openwheel T-Cam (t2v)",
              {"prompt": "a cozy felt-style race car speeding around a circuit, onboard T-cam "
                         "cockpit view looking ahead down the track, handmade wool texture, motion blur",
               "seed": 42},
              checks=DEFAULT_CHECKS, tags=["lora", "t2v", "combo"],
              required_loras=["cozy-felt-style", "openwheel-tcam-style"]))

# IC-LoRA. Catalog defaults skip_stage_2; `iclora_stage2_on` (E3) overrides that
# so both stages run. 1.3.0 no longer forces stage 2 onto streaming
# (`use_lora_in_stage_2` is gone). Low-VRAM streaming still hits the cache's
# non-cacheable eviction via `_is_streaming`.
# Scoped to what the app actually supports today:
#   - canny / depth: production control IC-LoRAs (api_types.ConditioningType).
#     Route: POST /api/extract-conditioning then POST /api/generate (ic_lora.py,
#     IcLoraGenerateRequest, conditioning_type=canny|depth).
#   - day-to-night: one transformation IC-LoRA (catalog id "day-to-night").
# Still needs_wiring: separate route + a source clip, not a plain payload override.
# Text-to-image (zit image model). Own route, no input asset.
_add(Scenario("t2i", "Text-to-Image",
              {"prompt": "a cozy felt duck in a sunlit room", "width": 1024, "height": 576,
               "numSteps": 4, "numImages": 1},
              route="/api/generate-image",
              checks=DEFAULT_CHECKS, tags=["t2i", "image"]))

# Extend / retake — own routes, driven by the felt-duck clip (VIDEO_ASSET).
_add(Scenario("video_extend", "Video Extend (end, +5s)",
              {"video_path": VIDEO_ASSET, "duration": 5.0,
               "prompt": "the duck suddenly flaps its wings and flies up out of the frame",
               "mode": "end"},
              route="/api/extend",
              checks=DEFAULT_CHECKS, tags=["extend"],
              requires_caps=["extend"]))
# Prepend note: extend(start) must END on the clip's first frame (duck already in
# scene, mid-jump), so the generated segment is anchored to a boundary that already
# CONTAINS the duck. An "entrance from an empty frame" is physically impossible here —
# the model interpolates backward from the duck-present boundary and just keeps the
# duck on screen. The prompt that works is a LEAD-IN / wind-up that flows into the
# clip's opening (duck at rest -> crouch -> spring), not an arrival from off-screen.
_add(Scenario("video_prepend", "Video Prepend (start, +5s)",
              {"video_path": VIDEO_ASSET, "duration": 5.0,
               "prompt": "the felt duck spins around in place, then crouches low and "
                         "springs upward, beginning to jump",
               "mode": "start"},
              route="/api/extend",
              checks=DEFAULT_CHECKS, tags=["extend"],
              requires_caps=["extend"]))
_add(Scenario("retake", "Retake / re-roll (2.0–5.0s)",
              {"video_path": VIDEO_ASSET, "start_time": 2.0, "duration": 3.0,
               "prompt": "the duck flaps hard and flies right out of the frame",
               "mode": "replace_audio_and_video"},
              route="/api/retake",
              checks=DEFAULT_CHECKS, tags=["retake"],
              requires_caps=["retake"]))

# IC-LoRA control (canny/depth): /api/ic-lora/generate re-derives conditioning from
# the source video internally. Requires the canny/depth IC-LoRA + depth models to be
# DOWNLOADED on the box, else the backend returns a clean 4xx (informative).
# canny follows sharp edges, depth follows blob masses — BOTH also condition on the
# source's BACKGROUND, so those background regions WILL be rendered as something. The
# fix is positive, not negative: describe the whole frame (foreground subject + a
# concrete background) so the model fills the background contours/blobs as intended
# scenery, instead of defaulting them to extra ducks (canny) or faces/heads (depth).
_add(Scenario("iclora_canny", "IC-LoRA control: canny",
              {"conditioning_type": "canny", "video_path": VIDEO_ASSET,
               "prompt": "a glossy yellow rubber duck bouncing on a smooth wooden tabletop, "
                         "a warm sunlit kitchen with soft blurred wooden shelves and potted plants "
                         "in the background, cozy morning light, cinematic product shot, high detail"},
              route="/api/ic-lora/generate",
              checks=DEFAULT_CHECKS, tags=["iclora", "control"],
              required_files=[CP_UNION_CONTROL],
              requires_caps=["builtin_control"]))
_add(Scenario("iclora_depth", "IC-LoRA control: depth",
              {"conditioning_type": "depth", "video_path": VIDEO_ASSET,
               "prompt": "a cute green frog hopping across a mossy rock, lush green garden foliage "
                         "and broad leaves filling the softly blurred background, gentle natural "
                         "daylight, shallow depth of field, high detail"},
              route="/api/ic-lora/generate",
              checks=DEFAULT_CHECKS, tags=["iclora", "control"],
              required_files=[CP_UNION_CONTROL, CP_DEPTH_PROCESSOR],
              requires_caps=["builtin_control"]))

# IC-LoRA transformation (catalog): day-to-night. Uses input_path (not video_path);
# requires the "day-to-night" IC-LoRA downloaded (else 409 IC_LORA_NOT_DOWNLOADED).
_add(Scenario("iclora_day_to_night", "IC-LoRA: Day to Night",
              {"ic_lora_id": "day-to-night", "conditioning_type": "custom",
               "input_path": VIDEO_ASSET, "prompt": "turn day into night"},
              route="/api/ic-lora/generate",
              checks=DEFAULT_CHECKS, tags=["iclora", "transform", "bump"],
              required_ic_loras=["day-to-night"]))

# LTX-2 bump gate (`python sanity.py --tags bump --gate-integrity`). E1/E2/E3/cancel/MKF.
# Do NOT run these under --fast: that forces 540p/5s and IC-LoRA resolution_factor=1.0,
# which hides the tiling/VRAM bugs the bump is checking.
_add(Scenario("t2v_540p_8s", "T2V 540p / 8s (E1 decode VRAM)",
              {"prompt": "a calm ocean at sunset", "seed": 42,
               "resolution": "540p", "duration": 8, "fps": 24},
              checks=BUMP_VIDEO_CHECKS, tags=["t2v", "resolution", "bump", "e1"]))
SCENARIOS["t2v_1080p_5s"].tags.append("bump")
SCENARIOS["t2v_1080p_5s"].tags.append("e1")
SCENARIOS["t2v_1080p_5s"].checks = list(BUMP_VIDEO_CHECKS)

# 5s @ 24fps = 120 frames. Same still at 0 / mid / last — exercises the Distilled
# guiding-latent swap (distilled_keyframe_guiding), not visual variety.
_add(Scenario("mkf_interpolation", "Multi-keyframe interpolation (guiding swap)",
              {"prompt": "a fox and a panda walking through a sunlit forest",
               "seed": 42, "resolution": "540p", "duration": 5, "fps": 24,
               "keyframes": [
                   {"imagePath": IMAGE_ASSET, "frameIndex": 0, "strength": 1.0},
                   {"imagePath": IMAGE_ASSET, "frameIndex": 60, "strength": 1.0},
                   {"imagePath": IMAGE_ASSET, "frameIndex": 119, "strength": 1.0},
               ]},
              checks=BUMP_VIDEO_CHECKS, tags=["t2v", "mkf", "bump"]))

# Catalog never sets skip_stage_2=false; E3 needs stage 2 actually running.
_add(Scenario("iclora_stage2_on", "IC-LoRA: Day to Night (stage 2 ON, E3)",
              {"ic_lora_id": "day-to-night", "conditioning_type": "custom",
               "input_path": VIDEO_ASSET, "prompt": "turn day into night",
               "skip_stage_2": False},
              route="/api/ic-lora/generate",
              checks=DEFAULT_CHECKS, tags=["iclora", "transform", "bump", "e3"],
              required_ic_loras=["day-to-night"]))

_add(Scenario("cancel_mid_denoise", "Cancel mid-denoise (t2v 540p/8s)",
              {"prompt": "a calm ocean at sunset", "seed": 42,
               "resolution": "540p", "duration": 8, "fps": 24},
              checks=[], tags=["t2v", "bump", "cancel"],
              cancel_after_s=4.0))

# Full tier of the live regression suite: existing scenarios worth running after a big
# change but too slow for the smoke tier (plain-LoRA adapters, portrait aspect ratio).
for _key in ("lora_cozy_felt", "lora_cozy_felt_openwheel", "t2v_720p_9x16"):
    SCENARIOS[_key].tags.append("full")

# Registers the smoke/full regression scenarios. Imported last: it needs the names above,
# and importing it first is safe (it imports this module back, which finishes first).
import regression_scenarios  # noqa: E402,F401
import regression_queued  # noqa: E402,F401


def ready() -> list[Scenario]:
    """Scenarios whose payloads are wired (safe to run today).

    Excludes the regression suite (tag ``regression``): it has its own tiers and would
    triple the plain sweep. Run it with ``--tags smoke`` / ``--tags smoke full bump``."""
    return [s for s in SCENARIOS.values() if not s.needs_wiring and "regression" not in s.tags]


def all_scenarios() -> list[Scenario]:
    return list(SCENARIOS.values())
