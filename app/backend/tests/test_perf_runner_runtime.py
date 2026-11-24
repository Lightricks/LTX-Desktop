"""Perf-runner Settings wiring (no live backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PERF = Path(__file__).resolve().parents[1] / "performance_runner"
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

import perf_config  # noqa: E402
import scenarios as scn  # noqa: E402

_VERSIONS = [
    {"model_id": "ltx-2.3-22b-distilled-1.1", "label": "2.3", "installed": True, "active": True},
    {"model_id": "ltx-2.3-22b-distilled", "label": "2.3 (1.0)", "installed": False, "active": False},
    {"model_id": "ltx-2.5-22b-distilled", "label": "2.5", "installed": True, "active": False},
]


def test_blocked_by_caps_follows_models_specs() -> None:
    caps_25 = {"family": "2.5", "retake": True, "extend": True, "builtin_control": True, "ic_lora": True}
    assert scn.SCENARIOS["retake"].blocked_by_caps(caps_25) is None
    assert scn.SCENARIOS["video_extend"].blocked_by_caps(caps_25) is None
    assert scn.SCENARIOS["iclora_canny"].blocked_by_caps(caps_25) is None
    assert scn.SCENARIOS["t2v_540p_8s"].blocked_by_caps(caps_25) is None
    assert scn.SCENARIOS["iclora_day_to_night"].blocked_by_caps(caps_25) is None
    assert scn.SCENARIOS["mkf_interpolation"].blocked_by_caps(caps_25) is None
    assert scn.SCENARIOS["retake"].blocked_by_caps(None) is None
    caps_off = {"family": "2.5", "retake": False, "extend": False, "builtin_control": True, "ic_lora": True}
    assert scn.SCENARIOS["retake"].blocked_by_caps(caps_off)
    assert scn.SCENARIOS["video_extend"].blocked_by_caps(caps_off)


def test_bump_scenarios_not_blocked_on_2_5() -> None:
    caps_25 = {"family": "2.5", "retake": True, "extend": True, "builtin_control": True, "ic_lora": True}
    blocked = [s.key for s in scn.all_scenarios() if "bump" in s.tags and s.blocked_by_caps(caps_25)]
    assert blocked == []


def test_activate_ltx_model_uses_installed_label(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[tuple[str, dict]] = []
    monkeypatch.setattr(perf_config, "_get", lambda path: {"versions": _VERSIONS})
    monkeypatch.setattr(perf_config, "_post", lambda path, payload: posted.append((path, payload)))
    assert perf_config.activate_ltx_model("2.5") == "ltx-2.5-22b-distilled"
    assert posted == [("/api/models/active-ltx-model", {"model_id": "ltx-2.5-22b-distilled"})]


def test_activate_ltx_model_rejects_uninstalled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perf_config, "_get", lambda path: {"versions": _VERSIONS})
    with pytest.raises(RuntimeError, match="not installed"):
        perf_config.activate_ltx_model("2.3 (1.0)")


def test_set_use_conv_vae_requires_2_5_and_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perf_config, "ltx_versions", lambda: _VERSIONS)  # active is 2.3
    with pytest.raises(RuntimeError, match="2.5"):
        perf_config.set_use_conv_vae(True)

    monkeypatch.setattr(
        perf_config, "ltx_versions",
        lambda: [{**_VERSIONS[2], "active": True}],
    )
    monkeypatch.setattr(perf_config, "described_vaes", lambda: {"diff": True, "conv": False})
    with pytest.raises(RuntimeError, match="conv VAE"):
        perf_config.set_use_conv_vae(True)
    posted: list[dict] = []
    monkeypatch.setattr(perf_config, "described_vaes", lambda: {"diff": True, "conv": True})
    monkeypatch.setattr(perf_config, "_post", lambda path, payload: posted.append(payload))
    perf_config.set_use_conv_vae(True)
    assert posted == [{"useConvVae": True}]


def test_cancel_waits_for_inference_phase_not_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    # status=running includes phase=starting (reservation). Cancel then would miss
    # the interrupt wrapper. after_s=0 so we cancel as soon as inference is seen.
    import time

    phases = iter(
        [
            {"status": "running", "phase": "starting"},
            {"status": "running", "phase": "starting"},
            {"status": "running", "phase": "inference"},
        ]
    )
    seen: list[str] = []

    def fake_get(path: str):
        if path != perf_config.STATUS_PATH:
            return {}
        st = next(phases)
        seen.append(str(st["phase"]))
        return st

    def fake_post(path: str, payload: dict) -> dict:
        if path == perf_config.GENERATE_PATH:
            time.sleep(1.5)
            return {"status": "cancelled"}
        if path == perf_config.CANCEL_PATH:
            assert seen[-1] == "inference", f"cancelled during {seen!r}"
            return {}
        return {}

    monkeypatch.setattr(perf_config, "_get", fake_get)
    monkeypatch.setattr(perf_config, "_post", fake_post)
    perf_config.trigger_generation_cancellable(
        after_s=0.0, payload=perf_config.GEN_PAYLOAD_TEMPLATE, wait_timeout_s=2.0
    )
    assert "inference" in seen


def test_cancel_fails_if_inference_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    posted: list[str] = []

    def fake_get(path: str):
        return {"status": "running", "phase": "starting"}

    def fake_post(path: str, payload: dict) -> dict:
        posted.append(path)
        if path == perf_config.GENERATE_PATH:
            time.sleep(2.0)
            return {"status": "complete"}
        return {}

    monkeypatch.setattr(perf_config, "_get", fake_get)
    monkeypatch.setattr(perf_config, "_post", fake_post)
    with pytest.raises(RuntimeError, match="inference"):
        perf_config.trigger_generation_cancellable(
            after_s=0.0, payload=perf_config.GEN_PAYLOAD_TEMPLATE, wait_timeout_s=0.3
        )
    assert perf_config.CANCEL_PATH in posted
