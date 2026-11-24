"""Wiring of the live regression suite (tiers, ordering, assets). No backend, no GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PERF = Path(__file__).resolve().parents[1] / "performance_runner"
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

import scenarios as scn  # noqa: E402


def _tier(*tags: str) -> list[scn.Scenario]:
    return sorted((s for s in scn.all_scenarios() if set(s.tags) & set(tags)), key=lambda s: s.order)


def test_plain_sweep_does_not_include_the_regression_suite() -> None:
    assert not any("regression" in s.tags for s in scn.ready())


def test_smoke_covers_every_flow_and_code_path() -> None:
    keys = {s.key for s in _tier("smoke")}
    assert keys >= {
        "smoke_t2v_270p",  # stage-1-only path
        "smoke_t2v_540p_2s",  # two-stage path
        "smoke_lora_t2v",  # plain LoRA in the request
        "smoke_mkf_270p",
        "smoke_mkf_540p",  # keyframe swap on each path
        "smoke_i2v_270p",
        "smoke_a2v_270p",
        "smoke_ia2v_270p",
        "smoke_t2i",
        "smoke_extend_end",
        "smoke_extend_start",
        "smoke_retake_av",
        "smoke_retake_video_only",
        "smoke_retake_audio_only",
        "smoke_retake_silent_source",
        "smoke_iclora_canny",
        "smoke_iclora_depth",
        "smoke_iclora_day_to_night",
        "smoke_cancel_extend",
        "smoke_t2v_270p_repeat",
        "smoke_q_t2v_270p",  # queued surface
        "smoke_q_i2v_end_270p",
        "smoke_q_a2v_270p",
        "smoke_q_extend_end",
        "smoke_q_retake_av",
        "smoke_q_batch_seeds",
        "smoke_q_cancel_extend",
    }


def test_full_tier_is_a_superset_of_smoke_and_the_bump_gate() -> None:
    full = {s.key for s in _tier("smoke", "full", "bump")}
    assert {s.key for s in _tier("smoke")} <= full
    assert {s.key for s in _tier("bump")} <= full
    assert {"full_cancel_a2v", "full_cancel_iclora", "lora_cozy_felt"} <= full


def test_smoke_lora_is_the_reference_request_plus_an_adapter(monkeypatch) -> None:
    import catalog

    lora, ref = scn.SCENARIOS["smoke_t2v_270p"], scn.SCENARIOS["smoke_lora_t2v"]
    assert ref.route is None and ref.required_loras == ["cozy-felt-style"]
    assert ref.overrides == lora.overrides  # only the adapter differs, so "differs" means it was applied
    assert ref.order > lora.order  # the reference is recorded before the comparison
    assert "differs_from[t2v_270p_seed42]" in [c.__name__ for c in ref.checks]

    monkeypatch.setattr(catalog, "lora_ref", lambda cid: f"loras/{cid}/x.safetensors")
    body = ref.build_overrides(dict(ref.overrides))
    assert body["loras"] == [{"ref": "loras/cozy-felt-style/x.safetensors", "scale": 1.0}]


def test_smoke_runs_small_generations_only() -> None:
    for s in _tier("smoke"):
        if s.route is None and s.cancel_after_s is None and not s.queued:
            assert s.overrides["duration"] <= 5, s.key
            assert s.overrides["resolution"] in {"270p", "540p"}, s.key


def test_reference_scenarios_run_before_the_scenarios_that_compare_to_them() -> None:
    order = [s.key for s in _tier("smoke")]
    assert order[:2] == ["smoke_t2v_270p", "smoke_t2v_270p_seed_b"]
    assert order[-1] == "smoke_t2v_270p_repeat"
    assert order.index("smoke_cancel_extend") > order.index("smoke_iclora_day_to_night")
    full_order = [s.key for s in _tier("smoke", "full", "bump")]
    assert full_order.index("lora_cozy_felt") < full_order.index("smoke_t2v_270p_repeat")


def test_every_scenario_asset_exists() -> None:
    missing = [
        (s.key, path)
        for s in _tier("smoke", "full")
        for path in [*s.input_paths(), *_paths_in(s.overrides)]
        if not os.path.exists(path)
    ]
    assert missing == []


def _paths_in(overrides: dict) -> list[str]:
    found = [v for v in overrides.values() if isinstance(v, str) and v.endswith((".mp4", ".png", ".jpg", ".wav"))]
    for kf in overrides.get("keyframes") or []:
        found.append(kf["imagePath"])
    return found


def test_generating_scenarios_assert_more_than_a_file_exists() -> None:
    for s in _tier("smoke"):
        if s.cancel_after_s is not None or s.key == "smoke_t2i":
            continue
        behavioural = [c for c in s.checks if c not in scn.DEFAULT_CHECKS]
        assert behavioural, f"{s.key} only checks that a file exists"


def test_every_queued_create_route_has_a_scenario() -> None:
    routes = {job.route for s in _tier("smoke", "full") for job in s.queued or []}
    assert routes == {
        "/api/generations/text-to-video",
        "/api/generations/image-to-video",
        "/api/generations/audio-to-video",
        "/api/generations/extend",
        "/api/generations/retake",
    }


def test_last_keyframe_index_is_the_final_frame_of_a_two_second_clip() -> None:
    import regression_scenarios as reg
    from frame_math import compute_num_frames

    assert reg.KEYFRAME_LAST == compute_num_frames(2, 24) - 1


def test_cancel_scenarios_run_no_output_checks() -> None:
    cancels = [s for s in scn.all_scenarios() if "regression" in s.tags and s.cancel_after_s is not None]
    assert cancels
    assert all(not s.checks for s in cancels)


def _run_sanity(monkeypatch, *argv: str, tools: bool = True) -> int:
    import perf_config
    import sanity

    monkeypatch.setattr(sanity.shutil, "which", lambda name: name if tools else None)
    monkeypatch.setattr(sanity.keep_awake, "start", lambda: None)  # never hold the dev machine awake from a test
    monkeypatch.setattr(sys, "argv", ["sanity.py", *argv])
    monkeypatch.setattr(perf_config, "wait_for_backend", lambda *a, **k: None)
    return sanity.main()


def test_an_empty_or_mistyped_selection_fails_instead_of_passing(monkeypatch) -> None:
    assert _run_sanity(monkeypatch, "--tags", "no-such-tag") == 1
    assert _run_sanity(monkeypatch, "--only", "no_such_scenario") == 1
    assert _run_sanity(monkeypatch, "--only", "smoke_t2v_270p", "no_such_scenario") == 1


def test_a_crashed_backend_aborts_the_sweep_instead_of_erroring_every_scenario(monkeypatch, tmp_path) -> None:
    import perf_config
    import sanity

    ran: list[str] = []

    def crashing_run(scenario, **_kwargs):
        ran.append(scenario.key)
        return {"key": scenario.key, "title": scenario.title, "status": "ERROR", "wall_s": 1.0,
                "detail": "RemoteDisconnected", "output": None, "inputs": []}

    def backend_down(_path):
        raise ConnectionRefusedError

    monkeypatch.setattr(sanity, "run_one", crashing_run)
    monkeypatch.setattr(sanity, "collect_results", lambda rows, ts: tmp_path)
    monkeypatch.setattr(perf_config, "_get", backend_down)

    assert _run_sanity(monkeypatch, "--only", "smoke_t2v_270p", "smoke_t2v_540p_2s", "smoke_i2v_270p") == 1
    assert ran == ["smoke_t2v_270p"]


def test_every_scenario_announces_its_position_in_the_sweep(monkeypatch, tmp_path, capsys) -> None:
    import perf_config
    import sanity

    def passing_run(scenario, **_kwargs):
        return {"key": scenario.key, "title": scenario.title, "status": "PASS", "wall_s": 1.0,
                "detail": "", "output": None, "inputs": []}

    monkeypatch.setattr(sanity, "run_one", passing_run)
    monkeypatch.setattr(sanity, "collect_results", lambda rows, ts: tmp_path)
    # smoke_extend_end is blocked by capabilities: skipped scenarios still take their place in the count.
    monkeypatch.setattr(perf_config, "scenario_caps", lambda: {"extend": False, "family": "test"})

    keys = ["smoke_t2v_270p", "smoke_extend_end", "smoke_i2v_270p"]
    assert _run_sanity(monkeypatch, "--only", *keys) == 0

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[sanity]")]
    announced = [ln for ln in lines if "Running test" in ln]
    assert announced == [f"[sanity] Running test {i}/3" for i in (1, 2, 3)]
    # The announcement comes right before the scenario's own run or skip line.
    for i, key in enumerate(keys):
        nxt = lines[lines.index(announced[i]) + 1]
        assert key in nxt, (key, nxt)


def test_regression_run_refuses_to_start_without_ffmpeg(monkeypatch, capsys) -> None:
    import sanity

    monkeypatch.setattr(sanity, "run_one", lambda *a, **k: pytest.fail("must not run any scenario"))

    assert _run_sanity(monkeypatch, "--only", "smoke_t2v_270p", tools=False) == 1
    assert "ffmpeg and ffprobe not on PATH" in capsys.readouterr().err
