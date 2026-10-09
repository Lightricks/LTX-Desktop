"""Live regression suite: run by hand after any big change (LTX-2 bump, pipeline or
runtime refactor, dependency upgrade) to confirm every core flow still generates AND
still behaves correctly. Not for CI: it needs real weights, a GPU, and a running backend.

Tiers (select with ``--tags``, or the dashboard's "regression" buttons):

* ``smoke``  one small generation per flow and code path, a few minutes each at most.
             ``python sanity.py --tags smoke``
* ``full``   smoke plus the slow scenarios (plain LoRAs, portrait, cancel on every route)
             and the existing full-size ``bump`` gate.
             ``python sanity.py --tags smoke full bump``

Sizes are small on purpose. Local t2v at 270p/360p runs stage 1 only (the skip-stage-2
path) while 540p and up runs the two-stage pipeline, so smoke covers one of each. Small
canvases hide tiling/VRAM problems; the full-size ``bump`` scenarios are for those.

Each scenario adds behavioural checks (see ``checks.py``) on top of "the file exists":
keyframes and audio must be honoured, a retake must keep what it was told to keep, and
the same seed must reproduce after all the other flows have run.
"""

from __future__ import annotations

import os

import checks as chk
from scenarios import (
    AUDIO_ASSET,
    CP_DEPTH_PROCESSOR,
    CP_UNION_CONTROL,
    IMAGE_ASSET,
    SCENARIOS,
    VIDEO_ASSET,
    Scenario,
    output_exists,
    output_nonempty,
)

_ASSETS = os.path.dirname(IMAGE_ASSET)
KEYFRAME_B = os.path.join(_ASSETS, "reference_keyframe_b.jpg")   # a different scene (felt duck)
VIDEO_SILENT = os.path.join(_ASSETS, "reference_video_silent.mp4")  # VIDEO_ASSET with no audio stream
VIDEO_2S = os.path.join(_ASSETS, "reference_video_2s.mp4")          # first 2 s: cheap IC-LoRA input

SMALL = {"resolution": "270p", "duration": 2, "fps": 24}   # stage-1-only local t2v path
TWO_STAGE = {"resolution": "540p", "duration": 2, "fps": 24}
LOW_RES = {"width": 576, "height": 320}                    # extend/retake/IC-LoRA output size
PROMPT = "a red sports car driving fast down a city street, camera tracking alongside"
KEYFRAME_LAST = 48                                          # 2 s @ 24 fps is 49 frames: index 48 is the last
REF = "t2v_270p_seed42"

# Run positions (Scenario.order): references first, cancels then the reproduction check last.
FIRST, CANCELS, LAST = -10, 5, 10


def add_regression(key: str, title: str, overrides: dict, checks: list, *, tiers: tuple[str, ...] = ("smoke",),
         tags: tuple[str, ...] = (), **kwargs) -> None:
    cancelling = kwargs.get("cancel_after_s") is not None  # a cancelled run has no output to check
    SCENARIOS[key] = Scenario(
        key, title, overrides,
        checks=[] if cancelling else [output_exists, output_nonempty, *checks],
        tags=["regression", *tiers, *tags],
        **kwargs,
    )


def _keyframes(last_image: str) -> list[dict]:
    return [
        {"imagePath": IMAGE_ASSET, "frameIndex": 0, "strength": 1.0},
        {"imagePath": last_image, "frameIndex": KEYFRAME_LAST, "strength": 1.0},
    ]


def keyframe_checks() -> list:
    return [
        chk.media_ok(duration=(1.5, 2.6)),
        chk.not_dead(),
        chk.frame_matches(IMAGE_ASSET, at="first"),
        chk.frame_matches(KEYFRAME_B, at="last"),
    ]


def a2v_checks(*, with_image: bool = False) -> list:
    """Output length follows the audio; the source speech is carried through."""
    checks = [chk.media_ok(audio=True, duration=(4.5, 6.0)), chk.not_dead(), chk.audio_matches(AUDIO_ASSET)]
    return [*checks, chk.frame_matches(IMAGE_ASSET, at="first")] if with_image else checks


def extend_checks(mode: str) -> list:
    """The source clip is kept at the boundary the new footage attaches to."""
    keep = {"at": "first", "reference_at": "first"} if mode == "end" else {"at": "last", "reference_at": "last"}
    return [chk.media_ok(audio=True, duration=(6.0, 9.0)), chk.not_dead(), chk.frame_matches(VIDEO_ASSET, **keep)]


def retake_checks(mode: str, source: str = VIDEO_ASSET) -> list:
    """Whatever the mode leaves alone must match the source: frames outside the window, the
    picture for an audio-only retake, the soundtrack for a video-only one."""
    checks = [chk.media_ok(duration=(4.4, 5.7)), chk.not_dead()]
    if mode == "replace_video":
        return [*checks, chk.media_ok(audio=True), chk.audio_matches(source)]
    if mode == "replace_audio":
        return [*checks, chk.media_ok(audio=True), chk.frame_matches(source, at=2.0, reference_at=2.0)]
    return [*checks, chk.frame_matches(source, at="first"), chk.frame_matches(source, at="last", reference_at="last")]


# -- text / image / audio -> video (default route) -------------------------------------
add_regression("smoke_t2v_270p", "T2V 270p / 2s (stage-1-only path; reference for reproduction)",
     {"prompt": PROMPT, "seed": 42, **SMALL},
     [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead(), chk.remember(REF)],
     tags=("t2v",), order=FIRST)
add_regression("smoke_t2v_270p_seed_b", "T2V 270p / 2s, different seed (seed is honoured)",
     {"prompt": PROMPT, "seed": 43, **SMALL},
     [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead(), chk.differs_from(REF)],
     tags=("t2v",), order=FIRST)
add_regression("smoke_lora_t2v", "T2V 270p / 2s with a style LoRA (same request as the reference: adapter changes the video)",
     {"prompt": PROMPT, "seed": 42, **SMALL},
     [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead(), chk.differs_from(REF)],
     tags=("t2v", "lora"), required_loras=["cozy-felt-style"])
add_regression("smoke_t2v_540p_2s", "T2V 540p / 2s (two-stage path)",
     {"prompt": PROMPT, "seed": 42, **TWO_STAGE},
     [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead()], tags=("t2v",))
add_regression("smoke_i2v_270p", "I2V 270p / 2s (first frame is the image)",
     {"prompt": "gentle zoom in", "seed": 42, "imagePath": IMAGE_ASSET, **SMALL},
     [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead(), chk.frame_matches(IMAGE_ASSET, at="first")],
     tags=("i2v",))
add_regression("smoke_mkf_270p", "Keyframes 270p: first and last frame honoured (stage-1-only swap)",
     {"prompt": "a fox and a panda walking through a sunlit forest", "seed": 42,
      "keyframes": _keyframes(KEYFRAME_B), **SMALL},
     keyframe_checks(), tags=("mkf",))
add_regression("smoke_mkf_540p", "Keyframes 540p: first and last frame honoured (two-stage swap)",
     {"prompt": "a fox and a panda walking through a sunlit forest", "seed": 42,
      "keyframes": _keyframes(KEYFRAME_B), **TWO_STAGE},
     keyframe_checks(), tags=("mkf",))
add_regression("smoke_a2v_270p", "A2V 270p / 5s (output carries the source speech)",
     {"prompt": "a cartoon character telling a joke, talking to camera", "seed": 42,
      "audio": True, "audioPath": AUDIO_ASSET, "resolution": "270p", "duration": 5},
     a2v_checks(), tags=("a2v", "speech"))
add_regression("smoke_ia2v_270p", "Image+A2V 270p / 5s (first frame is the image, speech kept)",
     {"prompt": "two cartoon animals chatting at a cafe table", "seed": 42, "audio": True,
      "imagePath": IMAGE_ASSET, "audioPath": AUDIO_ASSET, "resolution": "270p", "duration": 5},
     a2v_checks(with_image=True), tags=("ia2v", "speech"))

# -- image ------------------------------------------------------------------------------
add_regression("smoke_t2i", "Text-to-Image",
     {"prompt": "a cozy felt duck in a sunlit room", "width": 1024, "height": 576, "numSteps": 4, "numImages": 1},
     [], route="/api/generate-image", tags=("t2i", "image"))

# -- extend -----------------------------------------------------------------------------
_extend = {"video_path": VIDEO_ASSET, "duration": 2.0, "resolution": LOW_RES}
add_regression("smoke_extend_end", "Extend end +2s (source kept at the start)",
     {**_extend, "mode": "end", "prompt": "the duck flaps its wings and flies up out of the frame"},
     extend_checks("end"),
     route="/api/extend", tags=("extend",), requires_caps=["extend"])
add_regression("smoke_extend_start", "Extend start +2s (source kept at the end)",
     {**_extend, "mode": "start", "prompt": "the felt duck crouches low and springs upward"},
     extend_checks("start"),
     route="/api/extend", tags=("extend",), requires_caps=["extend"])

# -- retake: every mode, plus a source with no audio track ---------------------------------
def _retake(mode: str, source: str = VIDEO_ASSET) -> dict:
    return {"video_path": source, "start_time": 1.0, "duration": 2.0, "mode": mode,
            "prompt": "the duck flaps hard and flies right out of the frame", "resolution": LOW_RES}


add_regression("smoke_retake_av", "Retake 1-3s, audio+video (frames outside the window kept)",
     _retake("replace_audio_and_video"),
     retake_checks("replace_audio_and_video"),
     route="/api/retake", tags=("retake",), requires_caps=["retake"])
add_regression("smoke_retake_video_only", "Retake 1-3s, video only (soundtrack untouched)",
     _retake("replace_video"),
     retake_checks("replace_video"),
     route="/api/retake", tags=("retake",), requires_caps=["retake"])
add_regression("smoke_retake_audio_only", "Retake 1-3s, audio only (picture untouched)",
     _retake("replace_audio"),
     retake_checks("replace_audio"),
     route="/api/retake", tags=("retake",), requires_caps=["retake"])
add_regression("smoke_retake_silent_source", "Retake 1-3s on a clip with no audio track",
     _retake("replace_audio_and_video", VIDEO_SILENT),
     retake_checks("replace_audio_and_video", VIDEO_SILENT),
     route="/api/retake", tags=("retake", "silent"), requires_caps=["retake"])

# -- IC-LoRA ------------------------------------------------------------------------------
_iclora_checks = [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead()]
add_regression("smoke_iclora_canny", "IC-LoRA canny (2s clip)",
     {"conditioning_type": "canny", "video_path": VIDEO_2S, "resolution": LOW_RES,
      "prompt": "a glossy yellow rubber duck on a wooden tabletop, warm sunlit kitchen behind, cinematic"},
     _iclora_checks, route="/api/ic-lora/generate", tags=("iclora", "control"),
     required_files=[CP_UNION_CONTROL], requires_caps=["builtin_control"])
add_regression("smoke_iclora_depth", "IC-LoRA depth (2s clip)",
     {"conditioning_type": "depth", "video_path": VIDEO_2S, "resolution": LOW_RES,
      "prompt": "a green frog on a mossy rock, lush garden foliage in the blurred background"},
     _iclora_checks, route="/api/ic-lora/generate", tags=("iclora", "control"),
     required_files=[CP_UNION_CONTROL, CP_DEPTH_PROCESSOR], requires_caps=["builtin_control"])
add_regression("smoke_iclora_day_to_night", "IC-LoRA day-to-night (2s clip, skip-stage-2 tiled path)",
     {"ic_lora_id": "day-to-night", "conditioning_type": "custom", "input_path": VIDEO_2S,
      "prompt": "turn day into night", "resolution_factor": 1.0},
     _iclora_checks, route="/api/ic-lora/generate", tags=("iclora", "transform"),
     required_ic_loras=["day-to-night"])

# -- cancel, then prove the backend recovered ---------------------------------------------
# Cancel targets are large so they can never finish before the cancel lands. The next
# scenario (and finally the reproduction check) proves the backend is usable afterwards.
add_regression("smoke_cancel_extend", "Cancel an extend mid-denoise",
     {"video_path": VIDEO_ASSET, "duration": 8.0, "mode": "end", "prompt": "the duck runs away"},
     [], route="/api/extend", tags=("cancel", "extend"), requires_caps=["extend"],
     cancel_after_s=1.5, order=CANCELS)
add_regression("full_cancel_a2v", "Cancel an A2V mid-denoise",
     {"prompt": "a cartoon character telling a joke", "seed": 42, "audio": True, "audioPath": AUDIO_ASSET,
      "resolution": "540p", "duration": 10},
     [], tiers=("full",), tags=("cancel", "a2v"), cancel_after_s=3.0, order=CANCELS + 1)
add_regression("full_cancel_iclora", "Cancel an IC-LoRA mid-denoise",
     {"ic_lora_id": "day-to-night", "conditioning_type": "custom", "input_path": VIDEO_ASSET,
      "prompt": "turn day into night"},
     [], tiers=("full",), route="/api/ic-lora/generate", tags=("cancel", "iclora"),
     required_ic_loras=["day-to-night"], cancel_after_s=3.0, order=CANCELS + 2)

# -- last: identical request as the first scenario, after every other flow ------------------
add_regression("smoke_t2v_270p_repeat", "T2V 270p / 2s again (reproduces after all other flows)",
     {"prompt": PROMPT, "seed": 42, **SMALL},
     [chk.media_ok(duration=(1.5, 2.6)), chk.not_dead(), chk.reproduces(REF)],
     tags=("t2v",), order=LAST)
