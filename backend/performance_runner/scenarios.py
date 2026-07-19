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
from dataclasses import dataclass, field
from typing import Any, Callable

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

    def input_paths(self) -> list[str]:
        """On-disk input assets referenced by this scenario (for the results viewer)."""
        keys = ("imagePath", "audioPath", "video_path", "input_path", "source_video", "control_video_path")
        out = []
        for k in keys:
            v = self.overrides.get(k)
            if isinstance(v, str) and os.path.exists(v):
                out.append(v)
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


DEFAULT_CHECKS = [output_exists, output_nonempty]


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

# IC-LoRA — the streaming stage_2 path exercises the patch's NON-CACHEABLE bypass.
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
              checks=DEFAULT_CHECKS, tags=["extend"]))
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
              checks=DEFAULT_CHECKS, tags=["extend"]))
_add(Scenario("retake", "Retake / re-roll (2.0–5.0s)",
              {"video_path": VIDEO_ASSET, "start_time": 2.0, "duration": 3.0,
               "prompt": "the duck flaps hard and flies right out of the frame",
               "mode": "replace_audio_and_video"},
              route="/api/retake",
              checks=DEFAULT_CHECKS, tags=["retake"]))

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
              required_files=[CP_UNION_CONTROL]))
_add(Scenario("iclora_depth", "IC-LoRA control: depth",
              {"conditioning_type": "depth", "video_path": VIDEO_ASSET,
               "prompt": "a cute green frog hopping across a mossy rock, lush green garden foliage "
                         "and broad leaves filling the softly blurred background, gentle natural "
                         "daylight, shallow depth of field, high detail"},
              route="/api/ic-lora/generate",
              checks=DEFAULT_CHECKS, tags=["iclora", "control"],
              required_files=[CP_UNION_CONTROL, CP_DEPTH_PROCESSOR]))

# IC-LoRA transformation (catalog): day-to-night. Uses input_path (not video_path);
# requires the "day-to-night" IC-LoRA downloaded (else 409 IC_LORA_NOT_DOWNLOADED).
_add(Scenario("iclora_day_to_night", "IC-LoRA: Day to Night",
              {"ic_lora_id": "day-to-night", "conditioning_type": "custom",
               "input_path": VIDEO_ASSET, "prompt": "turn day into night"},
              route="/api/ic-lora/generate",
              checks=DEFAULT_CHECKS, tags=["iclora", "transform"],
              required_ic_loras=["day-to-night"]))

def ready() -> list[Scenario]:
    """Scenarios whose payloads are wired (safe to run today)."""
    return [s for s in SCENARIOS.values() if not s.needs_wiring]


def all_scenarios() -> list[Scenario]:
    return list(SCENARIOS.values())
