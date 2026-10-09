"""Queued-surface (Home/Explore, ``/api/generations/...``) twins of the regression suite.

Same intent, same assertions (``regression_scenarios`` check builders) as the GenSpace
scenarios, but driven through the queue: input files are ingested as assets, the job is
created and polled through the ledger, and cancel goes through ``/generations/{id}/cancel``.
That covers what the synchronous routes cannot: the ``Create*Request`` schemas, the edit
executors, ledger status, and jobs queued back to back.

Smoke holds one twin per flow (t2v, i2v with an end frame, A2V, extend, retake, a
multi-job batch, cancel). The remaining variants are in ``full`` to keep smoke short.
Queued IC-LoRA and text-to-image have no queued route, so they stay GenSpace-only.
"""

from __future__ import annotations

import checks as chk
from queued import QueuedJob
from regression_scenarios import (
    CANCELS,
    KEYFRAME_B,
    LOW_RES,
    PROMPT,
    SMALL,
    TWO_STAGE,
    VIDEO_SILENT,
    a2v_checks,
    add_regression,
    extend_checks,
    keyframe_checks,
    retake_checks,
)
from scenarios import AUDIO_ASSET, IMAGE_ASSET, VIDEO_ASSET

_T2V, _I2V, _A2V = "/api/generations/text-to-video", "/api/generations/image-to-video", "/api/generations/audio-to-video"
_EXTEND, _RETAKE = "/api/generations/extend", "/api/generations/retake"
_FULL = ("full",)
_VIDEO_2S = (1.5, 2.6)


def _video(size: dict, **params) -> dict:
    return {"prompt": PROMPT, "seed": 42, "aspectRatio": "16:9", **size, **params}


def _add(key: str, title: str, jobs: list[QueuedJob], checks: list, *, tags: tuple[str, ...],
         tiers: tuple[str, ...] = ("smoke",), **kwargs) -> None:
    add_regression(key, title, {}, checks, tiers=tiers, tags=("queued", *tags), queued=jobs, **kwargs)


def _t2v(size: dict, **params) -> QueuedJob:
    return QueuedJob(_T2V, _video(size, **params))


def _i2v(size: dict, end_frame: str | None = None) -> QueuedJob:
    inputs = {"startFrame": IMAGE_ASSET, **({"endFrame": end_frame} if end_frame else {})}
    return QueuedJob(_I2V, {**_video(size), "prompt": "gentle zoom in", "aspectRatio": "auto"}, inputs)


def _a2v(*, with_image: bool = False) -> QueuedJob:
    inputs = {"audio": AUDIO_ASSET, **({"startFrame": IMAGE_ASSET} if with_image else {})}
    return QueuedJob(_A2V, {"prompt": "a cartoon character telling a joke", "aspectRatio": "16:9",
                            "resolution": "270p", "fps": 24, "seed": 42}, inputs)


def _extend(mode: str) -> QueuedJob:
    return QueuedJob(_EXTEND, {"prompt": "the duck flaps its wings and flies away", "duration": 2.0, "mode": mode,
                               "resolution": LOW_RES, "seed": 42}, {"video": VIDEO_ASSET})


def _retake(mode: str, source: str = VIDEO_ASSET) -> QueuedJob:
    return QueuedJob(_RETAKE, {"prompt": "the duck flaps hard and flies right out of the frame", "startTime": 1.0,
                               "duration": 2.0, "mode": mode, "resolution": LOW_RES, "seed": 42}, {"video": source})


_t2v_checks = [chk.media_ok(duration=_VIDEO_2S), chk.not_dead()]

# -- smoke: one twin per flow -------------------------------------------------------------
# Not compared with GenSpace's smoke_t2v_270p: with local text encoding GenSpace rewrites the
# prompt and Explore sends it as typed, so the two surfaces run different prompts. Same-seed
# determinism on this surface is smoke_q_batch_seeds.
_add("smoke_q_t2v_270p", "Queued T2V 270p / 2s", [_t2v(SMALL)], _t2v_checks, tags=("t2v",))
_add("smoke_q_i2v_end_270p", "Queued I2V 270p with end frame (start and end frames honoured)",
     [_i2v(SMALL, end_frame=KEYFRAME_B)], keyframe_checks(), tags=("i2v", "mkf"))
_add("smoke_q_a2v_270p", "Queued A2V 270p (length follows the audio, speech carried)",
     [_a2v()], a2v_checks(), tags=("a2v", "speech"))
_add("smoke_q_extend_end", "Queued extend end +2s", [_extend("end")], extend_checks("end"), tags=("extend",),
     requires_caps=["extend"])
_add("smoke_q_retake_av", "Queued retake 1-3s, audio+video", [_retake("replace_audio_and_video")],
     retake_checks("replace_audio_and_video"), tags=("retake",), requires_caps=["retake"])
_add("smoke_q_batch_seeds", "Queued: three t2v jobs submitted together (seed A, B, A)",
     [_t2v(SMALL), _t2v(SMALL, seed=43), _t2v(SMALL)], _t2v_checks, tags=("t2v", "batch"),
     batch_checks=[chk.seed_batch()])
_add("smoke_q_cancel_extend", "Queued: cancel an extend through the ledger",
     [QueuedJob(_EXTEND, {"prompt": "the duck runs away", "duration": 8.0, "mode": "end", "seed": 42},
                {"video": VIDEO_ASSET})],
     [], tags=("cancel", "extend"), requires_caps=["extend"], cancel_after_s=1.5, order=CANCELS)

# -- full: the remaining variants -----------------------------------------------------------
_add("full_q_t2v_540p_2s", "Queued T2V 540p / 2s (two-stage path)", [_t2v(TWO_STAGE)], _t2v_checks,
     tiers=_FULL, tags=("t2v",))
_add("full_q_i2v_270p", "Queued I2V 270p (first frame is the image)", [_i2v(SMALL)],
     [*_t2v_checks, chk.frame_matches(IMAGE_ASSET, at="first")], tiers=_FULL, tags=("i2v",))
_add("full_q_i2v_end_540p", "Queued I2V 540p with end frame (two-stage keyframe swap)",
     [_i2v(TWO_STAGE, end_frame=KEYFRAME_B)], keyframe_checks(), tiers=_FULL, tags=("i2v", "mkf"))
_add("full_q_ia2v_270p", "Queued image+A2V 270p", [_a2v(with_image=True)], a2v_checks(with_image=True),
     tiers=_FULL, tags=("ia2v", "speech"))
_add("full_q_extend_start", "Queued extend start +2s", [_extend("start")], extend_checks("start"),
     tiers=_FULL, tags=("extend",), requires_caps=["extend"])
for _mode, _label in (("replace_video", "video only"), ("replace_audio", "audio only")):
    _add(f"full_q_retake_{_mode.removeprefix('replace_')}", f"Queued retake 1-3s, {_label}", [_retake(_mode)],
         retake_checks(_mode), tiers=_FULL, tags=("retake",), requires_caps=["retake"])
_add("full_q_retake_silent_source", "Queued retake 1-3s on a clip with no audio track",
     [_retake("replace_audio_and_video", VIDEO_SILENT)], retake_checks("replace_audio_and_video", VIDEO_SILENT),
     tiers=_FULL, tags=("retake", "silent"), requires_caps=["retake"])
