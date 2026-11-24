"""Motion Track Home recipe: path sampler, control clip rendering, and job execution."""

from __future__ import annotations

import json
from pathlib import Path

import av
import pytest
from PIL import Image

from api_motion_path import MOTION_PATH_MAX_PATHS, MotionPath
from api_types import (
    IcLoraGenerateCancelledResponse,
    IcLoraSynthesizedStoredParams,
    ImageAssetMetadata,
    ImageMeta,
    LoraEntry,
    parse_lora_catalog,
)
from frame_math import compute_num_frames
from handlers.base import resolve_models_dir
from runtime_config.ltx_capabilities import budget_pixels, local_caps
from runtime_config.model_download_specs import resolve_ic_lora_path
from services.features.ic_lora_recipes import CAMERA_LOCK_PROMPT_TEMPLATE, IC_LORA_RECIPES
from services.features.video.ic_lora_track_recipe import IcLoraTrackRecipeExecutor
from services.features.video.job_common import resolve_job_local_model_id
from services.features.video.motion_path import positions, render_motion_track
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.types import OutputAllocation, OutputPlan
from tests.fakes.generation import FakeJobSession
from services.records import AssetRecord, CapabilityFailedError, GenerationRecord
from tests.lora_catalog_helpers import download_spec

_WEIGHTS = "motion-track.safetensors"
_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"
_LOCAL_ENHANCER_OFFERING = "ltx-2.3-fast"
_GOLDEN = Path(__file__).resolve().parents[2] / "shared" / "motion-path-golden.json"
_SHARED = Path(__file__).resolve().parents[2] / "shared" / "motion-path.json"
_PATH: dict[str, object] = {
    "points": [
        {"x": 0.2, "y": 0.3, "handleOut": {"x": 0.1, "y": 0.0}},
        {"x": 0.8, "y": 0.7},
    ],
    "easing": "linear",
    "startHoldSec": 0,
    "endHoldSec": 0,
}


def _png(path: Path, size: tuple[int, int] = (64, 36)) -> Path:
    Image.new("RGB", size, color=(200, 30, 30)).save(path, format="PNG")
    return path


def _install_motion_track(fake_services, test_state) -> None:
    catalog = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
    shipped = next(
        entry
        for entry in parse_lora_catalog(catalog.read_text()).ic_loras
        if entry.id == "motion-track"
    )
    fake_services.lora_catalog_provider._catalog.ic_loras.append(
        shipped.model_copy(update={"download": download_spec(_WEIGHTS)})
    )
    path = resolve_ic_lora_path(
        resolve_models_dir(test_state.state, test_state.config), "motion-track", _WEIGHTS
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def _clip_frames(path: Path) -> list[tuple[int, int, tuple[int, int, int]]]:
    """Per frame: width, height and the pixel at the first path point."""
    out: list[tuple[int, int, tuple[int, int, int]]] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            red, green, blue = rgb[int(0.3 * height), int(0.2 * width)]
            out.append((width, height, (int(red), int(green), int(blue))))
    return out


def test_the_dot_follows_the_frames_the_ltx_io_sampler_makes() -> None:
    golden = json.loads(_GOLDEN.read_text())
    for case in golden["cases"]:
        frames = positions(
            MotionPath.model_validate(case["path"]),
            duration=case["duration"],
            frames=case["frames"],
        )
        assert len(frames) == case["frames"], case["name"]
        # The golden file keeps every step-th frame. The frame count is 8k+1, so the last stays.
        kept = frames[:: golden["step"]]
        for (x, y), (want_x, want_y) in zip(kept, case["positions"], strict=True):
            assert (x, y) == pytest.approx((want_x, want_y), abs=1e-9), case["name"]


def test_the_shared_example_path_is_a_path_the_backend_accepts() -> None:
    # The Reuse example fills the form from this file. A schema change that rejects it would
    # fail every first Generate from the example.
    MotionPath.model_validate(json.loads(_SHARED.read_text())["examplePath"])


def test_the_control_clip_has_every_output_frame_and_the_dot_where_the_path_starts_and_ends(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "track.mp4"
    render_motion_track(
        [MotionPath.model_validate(_PATH)], width=320, height=180, duration=2, fps=24, dest=clip
    )
    frames = _clip_frames(clip)
    assert len(frames) == compute_num_frames(2, 24)
    assert (frames[0][0], frames[0][1]) == (320, 180)
    _, _, (red, _, blue) = frames[0]
    assert blue > 150 and red < 100
    # At the end the dot has left the start point, so the pixel there is black again.
    assert max(frames[-1][2]) < 60


def test_the_control_clip_has_one_dot_for_each_path(tmp_path: Path) -> None:
    other = {**_PATH, "points": [{"x": 0.2, "y": 0.8}, {"x": 0.8, "y": 0.2}]}
    clip = tmp_path / "tracks.mp4"
    render_motion_track(
        [MotionPath.model_validate(_PATH), MotionPath.model_validate(other)],
        width=320,
        height=180,
        duration=2,
        fps=24,
        dest=clip,
    )
    with av.open(str(clip)) as container:
        first = next(container.decode(video=0)).to_ndarray(format="rgb24")
    # Frame 0 holds both starts, (0.2, 0.3) and (0.2, 0.8). Between them the frame is black.
    for y in (0.3, 0.8):
        assert first[int(y * 180), int(0.2 * 320)][2] > 150
    assert first[int(0.55 * 180), int(0.2 * 320)].max() < 60


def _dot_rgb(track: int) -> tuple[int, int, int]:
    color = json.loads(_SHARED.read_text())["dotColors"][track]
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _is_the_dot(pixel: np.ndarray, track: int = 0) -> bool:
    return all(abs(int(got) - want) < 40 for got, want in zip(pixel, _dot_rgb(track), strict=True))


def test_every_path_has_its_own_dot_color() -> None:
    colors = json.loads(_SHARED.read_text())["dotColors"]
    assert len(set(colors)) == len(colors) >= MOTION_PATH_MAX_PATHS


def _clip_rgb_frames(clip: Path) -> list[np.ndarray]:
    with av.open(str(clip)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_each_track_in_the_control_clip_keeps_its_own_pace(tmp_path: Path) -> None:
    # The second track holds for 1 s of the 2 s clip. At the first quarter the first track has
    # left its start. The second track still has its dot on its start.
    held = {**_PATH, "points": [{"x": 0.2, "y": 0.8}, {"x": 0.8, "y": 0.8}], "startHoldSec": 1}
    first = MotionPath.model_validate(_PATH)
    clip = tmp_path / "paced.mp4"
    render_motion_track(
        [first, MotionPath.model_validate(held)],
        width=320,
        height=180,
        duration=2,
        fps=24,
        dest=clip,
    )
    frames = _clip_rgb_frames(clip)
    index = len(frames) // 4
    x, y = positions(first, duration=2, frames=len(frames))[index]
    assert _is_the_dot(frames[index][int(y * 180), int(x * 320)], 0)
    assert _is_the_dot(frames[index][int(0.8 * 180), int(0.2 * 320)], 1)


def test_each_dot_has_the_color_of_its_track_where_the_tracks_cross(tmp_path: Path) -> None:
    # Two tracks that cross. Each dot keeps the color of its own track.
    down = MotionPath.model_validate({**_PATH, "points": [{"x": 0.2, "y": 0.2}, {"x": 0.8, "y": 0.8}]})
    up = MotionPath.model_validate({**_PATH, "points": [{"x": 0.2, "y": 0.8}, {"x": 0.8, "y": 0.2}]})
    clip = tmp_path / "cross.mp4"
    render_motion_track([down, up], width=640, height=360, duration=4, fps=24, dest=clip)
    frames = _clip_rgb_frames(clip)
    index = 3 * len(frames) // 4
    for track, path in enumerate((down, up)):
        x, y = positions(path, duration=4, frames=len(frames))[index]
        assert _is_the_dot(frames[index][int(y * 360), int(x * 640)], track)


class _ClipRecorder:
    """Stands in for the pipeline. Reads the control clip while it exists."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.request: object | None = None
        self.clip_path: Path | None = None
        self.clip_frames: list[tuple[int, int, tuple[int, int, int]]] = []

    def generate_local_reserved(self, request: object, **_kwargs: object) -> object:
        self.request = request
        self.clip_path = Path(str(getattr(request, "input_path")))
        self.clip_frames = _clip_frames(self.clip_path)
        if self.fail:
            raise RuntimeError("pipeline failed")
        return IcLoraGenerateCancelledResponse(status="cancelled")


class _Assets:
    def __init__(self, asset: AssetRecord) -> None:
        self._asset = asset

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        return self._asset if asset_id == self._asset.id else None


def _track_job(
    tmp_path: Path,
    *,
    model: str = "ltx-2.5-fast",
    resolution: str = "540p",
) -> tuple[GenerationRecord, _Assets, AssetRecord, tuple[OutputAllocation, ...]]:
    image = _png(tmp_path / "a.png")
    asset = AssetRecord(
        id="img",
        media_kind="image",
        origin="uploaded",
        path=str(image),
        mime_type="image/png",
        name=image.name,
        metadata=ImageAssetMetadata(mediaType="image", metadata=ImageMeta(width=64, height=36)),
        created_at=0,
    )
    params = IcLoraSynthesizedStoredParams(
        prompt="  the frog conducts  ",
        model=model,
        resolution=resolution,
        audioMode="generated",
        aspectRatio="16:9",
        duration=2,
        fps=24,
        loras=[LoraEntry(ref="", scale=0.8, catalogId="motion-track")],
    )
    generation = GenerationRecord(
        id="gen",
        feature="motion-track",
        contract_version=1,
        status="running",
        spec={
            "params": params.model_dump(mode="json"),
            "inputs": {"image": {"assetId": asset.id}, "paths": [_PATH]},
        },
        created_at=0,
        queued_at=0,
        attempt_count=1,
        started_at=0,
        outputs=(),
    )
    outputs = (
        OutputAllocation(
            plan=OutputPlan(slot="output", media_kind="video", mime_type="video/mp4", name="o.mp4"),
            asset_id="out",
            dest_path=str(tmp_path / "output.mp4"),
        ),
    )
    return generation, _Assets(asset), asset, outputs


def test_run_sends_the_camera_locked_scene_without_a_rewrite_and_the_clip_at_the_denoise_size(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.prompt_enhancer_enabled = True
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "REWRITTEN WITHOUT LOCK"
    _install_motion_track(fake_services, test_state)

    generation, assets, asset, outputs = _track_job(tmp_path, model=_LOCAL_ENHANCER_OFFERING)
    models_dir = test_state.config.default_models_dir
    executor = IcLoraTrackRecipeExecutor(
        test_state.ic_lora, assets, lambda: models_dir, recipe=IC_LORA_RECIPES["motion-track"]
    )
    executor.execute(generation, outputs, FakeJobSession())

    expected_prompt = CAMERA_LOCK_PROMPT_TEMPLATE.replace("{prompt}", "the frog conducts")
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == expected_prompt
    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
    assert fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls == []

    recorder = _ClipRecorder()
    with pytest.raises(GenerationCancelledError):
        IcLoraTrackRecipeExecutor(
            recorder, assets, lambda: models_dir, recipe=IC_LORA_RECIPES["motion-track"]
        ).execute(generation, outputs, FakeJobSession())
    request = recorder.request
    assert getattr(request, "prompt") == expected_prompt
    assert getattr(request, "prompt_provenance") == "typed"
    assert [image.path for image in getattr(request, "images")] == [asset.path]
    size = budget_pixels(
        local_caps(resolve_job_local_model_id(models_dir, _LOCAL_ENHANCER_OFFERING)),
        "540p",
        "16:9",
        mode="video",
    )
    assert len(recorder.clip_frames) == compute_num_frames(2, 24)
    assert (recorder.clip_frames[0][0], recorder.clip_frames[0][1]) == size


@pytest.mark.parametrize("fail", [False, True], ids=["cancelled", "failed"])
def test_run_deletes_the_control_clip_when_the_run_is_cancelled_or_fails(
    test_state, create_fake_model_files, tmp_path: Path, fail: bool
) -> None:
    create_fake_model_files()
    generation, assets, _, outputs = _track_job(tmp_path)
    recorder = _ClipRecorder(fail=fail)
    models_dir = test_state.config.default_models_dir
    executor = IcLoraTrackRecipeExecutor(
        recorder, assets, lambda: models_dir, recipe=IC_LORA_RECIPES["motion-track"]
    )
    with pytest.raises(RuntimeError if fail else GenerationCancelledError):
        executor.execute(generation, outputs, FakeJobSession())
    assert recorder.clip_path is not None and not recorder.clip_path.exists()


def test_a_stored_high_res_spec_fails_without_the_recipe_slug_in_the_message(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    generation, assets, _, outputs = _track_job(tmp_path, resolution="2160p")
    models_dir = test_state.config.default_models_dir
    executor = IcLoraTrackRecipeExecutor(
        _ClipRecorder(), assets, lambda: models_dir, recipe=IC_LORA_RECIPES["motion-track"]
    )
    with pytest.raises(CapabilityFailedError) as raised:
        executor.execute(generation, outputs, FakeJobSession())
    assert "cannot render above 1080p" in str(raised.value)
    assert "motion-track" not in str(raised.value)

