"""IC-LoRA list-time drops ride with refusals; upload preflight verdicts real pixels."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api_types import (
    DownloadSpec,
    DownloadVariant,
    IcLoraCatalogItem,
    IcLoraControl,
    IcLoraSettings,
    InputSpec,
    InstructionSection,
    PreprocessingStep,
)
from frame_math import compute_num_frames
from runtime_config.budget_admission import unmeasured_refusal_message
from runtime_config.ic_lora_job_budget import with_budgeted_duration_controls
from runtime_config.video_job_budget import (
    VIDEO_JOB_TOO_LARGE,
    LocalJobFilter,
    build_local_job_filter,
)
from tests.fakes.services import FakeCapture
from tests.http_error_assertions import assert_http_error

_GIB = 1024**3


def _configure_windows(test_state, *, vram_gib: float, total_ram_gib: float) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"


def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    Image.new("RGB", size, color=(1, 2, 3)).save(path, format="PNG")
    return path


def _duration_item(
    options: list[int], *, skip_stage_2: bool, fps: int = 24
) -> IcLoraCatalogItem:
    return IcLoraCatalogItem(
        id="probe-entry",
        name="Probe",
        description="d",
        download=DownloadSpec(
            repo_id="org/x",
            variants=[
                DownloadVariant(
                    id="default",
                    label="Default",
                    filename="x.safetensors",
                    size_bytes=10,
                    base_model="LTX-2.3",
                )
            ],
        ),
        requires_hf_login=False,
        input=InputSpec(kind="image"),
        preprocessing=[
            PreprocessingStep(utility="image_to_frames", params={"fps": fps})
        ],
        instructions=[
            InstructionSection(kind="summary", title="What it does", body="b")
        ],
        controls=[
            IcLoraControl(
                id="duration",
                label="Duration",
                kind="int",
                default=5,
                options=list(options),
            )
        ],
        default_settings=IcLoraSettings(
            skip_stage_2=skip_stage_2, resolution_factor=1.5
        ),
    )


def _filter(test_state) -> LocalJobFilter:
    # model_id=None: the Windows IC-LoRA gate does not consult the model.
    # Unmeasured probes (None): free VRAM falls back to total, matching the
    # no-CUDA CI host the live probes used to read — deterministically.
    return build_local_job_filter(
        test_state.config,
        model_id=None,
        free_vram_bytes=None,
        mps_pool_bytes=None,
    )


def _duration_control(item: IcLoraCatalogItem) -> IcLoraControl:
    return next(c for c in item.controls if c.id == "duration")


def test_dropped_two_stage_duration_past_the_ceiling_carries_curve_figures(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=16, total_ram_gib=64)
    job_filter = _filter(test_state)

    fenced = with_budgeted_duration_controls(
        _duration_item([5, 20], skip_stage_2=False), job_filter=job_filter
    )
    control = _duration_control(fenced)
    assert control.options == [5]
    assert control.dropped_options is not None
    assert [(d.value, d.reason) for d in control.dropped_options] == [
        (20, "stream_over")
    ]
    assert control.dropped_options[0].over_gib is not None

    budgeted = with_budgeted_duration_controls(
        _duration_item([5, 20], skip_stage_2=True), job_filter=job_filter
    )
    control = _duration_control(budgeted)
    assert control.options == [5]
    assert control.dropped_options is not None
    assert [(d.value, d.reason) for d in control.dropped_options] == [
        (20, "stream_over")
    ]
    assert control.dropped_options[0].message == (
        "This job needs 25.9 GB of GPU memory but this card has 16.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    )


def test_all_dropped_keeps_full_options_with_advisory_refusals(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=7.9, total_ram_gib=64)
    out = with_budgeted_duration_controls(
        _duration_item([5, 20], skip_stage_2=True),
        job_filter=_filter(test_state),
    )
    control = _duration_control(out)
    assert control.options == [5, 20]
    assert control.dropped_options is not None
    assert [(d.value, d.reason) for d in control.dropped_options] == [
        (5, "stream_over"),
        (20, "stream_over"),
    ]
    assert control.dropped_options[0].message == (
        "This job needs 11.6 GB of GPU memory but this card has 7.9 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    )


def test_unsupported_empties_options_with_every_option_dropped(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=31, total_ram_gib=64)
    test_state.config.local_generations_mode = "unsupported"
    out = with_budgeted_duration_controls(
        _duration_item([5, 8], skip_stage_2=True),
        job_filter=_filter(test_state),
    )
    control = _duration_control(out)
    assert control.options == []
    assert control.dropped_options is not None
    assert [(d.value, d.reason) for d in control.dropped_options] == [
        (5, "unsupported"),
        (8, "unsupported"),
    ]
    assert control.dropped_options[0].message == (
        "This machine cannot run local generation."
    )


def test_preflight_cell_mirrors_generate_probe_422(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=12, total_ram_gib=64)
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = _write_png(tmp_path / "in.png", size=(768, 1280))

    preflight = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "durations": [5, 20],
        },
    )
    assert preflight.status_code == 200
    cells = {c["duration"]: c for c in preflight.json()["cells"]}
    assert cells[5]["ok"] is True
    assert cells[20]["ok"] is False

    generate = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "control_values": {"duration": 20},
            "prompt": "a slow orbit",
            "conditioning_type": "custom",
        },
    )
    assert generate.status_code == 422
    payload = generate.json()
    assert payload["code"] == VIDEO_JOB_TOO_LARGE
    assert cells[20]["message"] == payload["message"]
    assert cells[20]["code"] == payload["code"]


def test_preflight_video_kind_verdicts_whole_clip_once(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=12, total_ram_gib=64)
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    frames = compute_num_frames(10, 24)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * frames, fps=24, width=960, height=540)
    )

    preflight = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "durations": [5, 20],
        },
    )
    assert preflight.status_code == 200
    cells = preflight.json()["cells"]
    assert len(cells) == 1
    assert cells[0]["duration"] == frames / 24
    assert cells[0]["ok"] is True

    generate = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert generate.status_code == 200


def test_preflight_unknown_entry_404s_like_generate(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=12, total_ram_gib=64)
    img = _write_png(tmp_path / "in.png")
    resp = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "no-such-entry",
            "input_path": str(img),
            "durations": [5],
        },
    )
    assert_http_error(resp, status_code=404, code="UNKNOWN_IC_LORA")


def test_preflight_input_errors_match_generate(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=12, total_ram_gib=64)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 100)

    mismatch = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(clip),
            "durations": [5],
        },
    )
    assert_http_error(
        mismatch,
        status_code=400,
        code="HTTP_400",
        message="This IC-LoRA expects image input, but got '.mp4'",
    )

    missing = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(tmp_path / "gone.png"),
            "durations": [5],
        },
    )
    assert missing.status_code == 400
    assert "Input not found" in missing.json()["message"]


def test_preflight_two_stage_past_the_ceiling_is_refused_untested_in_cautious(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
) -> None:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=31, total_ram_gib=64)
    test_state.state.app_settings.budget_mode = "cautious"
    img = _write_png(tmp_path / "in.png", size=(1024, 1024))
    resp = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "durations": [20],
            "skip_stage_2": False,
        },
    )
    assert resp.status_code == 200
    (cell,) = resp.json()["cells"]
    assert cell["ok"] is False
    assert cell["message"] == unmeasured_refusal_message("two_stage_range")
    assert cell["code"] == VIDEO_JOB_TOO_LARGE
    assert cell["tier"] == "unmeasured"


def test_small_source_passes_list_time_dropped_duration(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
) -> None:
    create_fake_model_files()
    # 12 sits between the list-time 5s ask (11.00) and 8s ask (13.87): the list
    # keeps 5s and drops 8s/20s, while the upload passes 8s on real pixels.
    _configure_windows(test_state, vram_gib=12, total_ram_gib=64)

    listed = client.get("/api/ic-loras")
    assert listed.status_code == 200
    entry = next(
        item["ic_lora"]
        for item in listed.json()["ic_loras"]
        if item["ic_lora"]["id"] == "ingredients-v1"
    )
    control = next(c for c in entry["controls"] if c["id"] == "duration")
    assert control["options"] == [5]
    assert control["dropped_options"] is not None
    assert [d["value"] for d in control["dropped_options"]] == [8, 20]

    img = _write_png(tmp_path / "tiny.png", size=(256, 256))
    preflight = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "durations": [5, 8, 20],
        },
    )
    assert preflight.status_code == 200
    cells = preflight.json()["cells"]
    # 8s flips from list-dropped to passing on real pixels (5s passes both);
    # 20s still fails because duration scales tokens even on the bucket-minimum canvas.
    assert [c["ok"] for c in cells] == [True, True, False]
