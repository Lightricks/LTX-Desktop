"""A clean local run proves its own shape on this device, and nothing wider."""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from runtime_config.budget_evidence import JobShape
from runtime_config.device_calibration import (
    CALIBRATION_EPOCH,
    EMPTY_CALIBRATION,
    CalibrationBucket,
    DeviceCalibration,
    DeviceIdentity,
    EnvelopePoint,
    FrontierPoint,
    JobEnvelope,
    RunMemoryBaseline,
    RunMemoryEnd,
    clean_run_verdict,
    envelope_for,
    merge_frontier,
)
from runtime_config.eta import estimate_envelope_wall_time
from runtime_config.video_job_budget import VIDEO_JOB_TOO_LARGE
from services.gpu_info import run_peak
from services.records import UnavailableError
from services.sqlite_store import SqliteStore

_GIB = 1024**3
_MIB = 1024**2
_5090 = DeviceIdentity(
    system="Windows", gpu_model="NVIDIA GeForce RTX 5090", total_memory_bytes=int(31.84 * _GIB)
)
_MAC = DeviceIdentity(system="Darwin", gpu_model="Apple M4 Max", total_memory_bytes=128 * _GIB)
_BUCKET = CalibrationBucket(
    model_id="ltx-2.5-22b-distilled",
    feature="fast",
    adapter_id="",
    stage_mode="",
    load_mode="streaming_models_loading",
    tiled=False,
    chunked=False,
)
_UNMEASURED_1440P = {"prompt": "test", "resolution": "1440p", "model": "fast", "duration": 20, "fps": 50}
_START = RunMemoryBaseline(
    free_vram_bytes=int(24.0 * _GIB), reserved_bytes=int(0.5 * _GIB), swap_used_bytes=2 * _GIB
)


def _point(seq: float, canvas_px: int, windows: int = 1) -> FrontierPoint:
    return FrontierPoint(
        point=EnvelopePoint(seq=seq, canvas_px=canvas_px, windows=windows),
        peak_growth_bytes=20 * _GIB,
    )


def _job(seq: float, canvas_px: int, windows: int = 1) -> JobEnvelope:
    return JobEnvelope(
        bucket=_BUCKET,
        point=EnvelopePoint(seq=seq, canvas_px=canvas_px, windows=windows),
        priced_gib=27.0,
    )


def _calibration(*runs: FrontierPoint) -> DeviceCalibration:
    frontier: tuple[FrontierPoint, ...] = ()
    for run in runs:
        frontier = merge_frontier(frontier, run) or frontier
    return DeviceCalibration(identity=_5090, frontier={_BUCKET: frontier})


def _widen(store: SqliteStore, run: FrontierPoint) -> tuple[FrontierPoint, ...] | None:
    return store.widen_device_frontier(
        _5090, CALIBRATION_EPOCH, _BUCKET, run, priced_gib=27.0, generation_id="g"
    )


def _end(peak_gib: float, *, min_free_gib: float = 4.0) -> RunMemoryEnd:
    return RunMemoryEnd(
        peak_reserved_bytes=int(peak_gib * _GIB),
        min_free_vram_bytes=int(min_free_gib * _GIB),
        swap_used_bytes=2 * _GIB,
    )


def _configure_5090(test_state, fake_services, create_fake_model_files) -> None:
    create_fake_model_files()
    test_state.config.system = "Windows"
    test_state.config.gpu_device_name = _5090.gpu_model
    test_state.config.vram_gb = 31
    test_state.config.vram_bytes = _5090.total_memory_bytes
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    test_state.state.app_settings.use_local_text_encoder = True
    gpu = fake_services.gpu_info
    gpu.free_vram_bytes = _START.free_vram_bytes
    gpu.reserved_bytes = _START.reserved_bytes
    gpu.swap_used_bytes = _START.swap_used_bytes
    gpu.run_end = _end(peak_gib=20.0)


def _post_unmeasured(client, test_state, mode: str) -> int:
    test_state.state.app_settings.budget_mode = mode
    return client.post("/api/generate", json=_UNMEASURED_1440P).status_code


def test_two_incomparable_runs_do_not_cover_their_corner() -> None:
    long_1080p = _point(seq=60_000, canvas_px=1920 * 1088)
    short_4k = _point(seq=30_000, canvas_px=3840 * 2176)
    calibration = _calibration(long_1080p, short_4k)

    assert calibration.covering(_job(60_000, 3840 * 2176)) is None
    assert calibration.covering(_job(50_000, 1920 * 1088)) == long_1080p
    assert calibration.covering(_job(25_000, 2560 * 1472)) == short_4k


def test_equal_shape_rerun_graduates() -> None:
    ran = _point(seq=60_000, canvas_px=2560 * 1472, windows=2)

    assert _calibration(ran).covering(_job(60_000, 2560 * 1472, windows=2)) == ran


def test_dominated_run_adds_no_row(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    big = _point(seq=60_000, canvas_px=2560 * 1472)

    assert _widen(store, big) == (big,)
    assert _widen(store, _point(seq=40_000, canvas_px=1920 * 1088)) is None
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {_BUCKET: (big,)}


def test_dominating_run_prunes_only_points_within_it(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    small = _point(seq=30_000, canvas_px=1920 * 1088)
    neighbor = _point(seq=20_000, canvas_px=3840 * 2176)
    dominating = _point(seq=60_000, canvas_px=2560 * 1472)
    _widen(store, small)
    _widen(store, neighbor)

    assert _widen(store, dominating) == (dominating, neighbor)
    loaded = store.load_device_frontier(_5090, CALIBRATION_EPOCH)
    assert set(loaded[_BUCKET]) == {dominating, neighbor}


def test_run_whose_reserved_growth_exceeds_start_free_is_not_recorded() -> None:
    # The 1080p retake that paged 0.4 GiB past free and still finished.
    assert clean_run_verdict(_5090, _START, _end(peak_gib=25.0)) == "vram_growth"
    assert clean_run_verdict(_5090, _START, _end(peak_gib=23.0)) == "clean"


def test_run_that_hit_free_floor_is_not_recorded() -> None:
    squeezed = _end(peak_gib=20.0, min_free_gib=0.25)

    assert clean_run_verdict(_5090, _START, squeezed) == "free_floor"
    assert clean_run_verdict(_5090, _START, _end(peak_gib=20.0, min_free_gib=1.0)) == "clean"


def test_darwin_run_with_swap_growth_is_not_recorded() -> None:
    start = RunMemoryBaseline(free_vram_bytes=None, reserved_bytes=None, swap_used_bytes=_GIB)
    stalled = RunMemoryEnd(
        peak_reserved_bytes=None, min_free_vram_bytes=None, swap_used_bytes=_GIB + 512 * _MIB
    )
    steady = RunMemoryEnd(
        peak_reserved_bytes=None, min_free_vram_bytes=None, swap_used_bytes=_GIB + 100 * _MIB
    )

    assert clean_run_verdict(_MAC, start, stalled) == "swap_growth"
    assert clean_run_verdict(_MAC, start, steady) == "clean"


def test_cuda_run_with_swap_growth_is_not_recorded() -> None:
    # A host-RAM-shortfall run on Windows/Linux can stay within VRAM limits while
    # paging host RAM to swap during CPU offload.
    paged_swap = dataclasses.replace(_end(peak_gib=20.0), swap_used_bytes=2 * _GIB + 512 * _MIB)
    steady_swap = dataclasses.replace(_end(peak_gib=20.0), swap_used_bytes=2 * _GIB + 100 * _MIB)

    assert clean_run_verdict(_5090, _START, paged_swap) == "swap_growth"
    assert clean_run_verdict(_5090, _START, steady_swap) == "clean"


def test_cuda_run_without_swap_probe_still_graduates_on_vram_signals() -> None:
    start_no_swap = dataclasses.replace(_START, swap_used_bytes=None)
    end_no_swap = dataclasses.replace(_end(peak_gib=20.0), swap_used_bytes=None)

    assert clean_run_verdict(_5090, start_no_swap, end_no_swap) == "clean"


def test_missing_cuda_peak_probe_records_nothing() -> None:
    no_peak = RunMemoryEnd(
        peak_reserved_bytes=None, min_free_vram_bytes=4 * _GIB, swap_used_bytes=2 * _GIB
    )

    assert clean_run_verdict(_5090, _START, no_peak) == "no_evidence"
    assert clean_run_verdict(_5090, _START, _end(peak_gib=20.0)) == "clean"


def test_cancelled_generation_records_nothing(
    test_state, fake_services, create_fake_model_files
) -> None:
    _configure_5090(test_state, fake_services, create_fake_model_files)
    test_state.pipelines.load_gpu_pipeline("fast")
    store = SqliteStore(test_state.config.app_data_dir)
    ran = _job(60_000, 2560 * 1472)

    assert test_state.generation.try_reserve_generation_start()
    test_state.generation.start_generation("cancelled")
    assert run_peak._watch.active is True
    test_state.generation.cancel_generation()
    assert run_peak._watch.active is False
    test_state.generation.complete_generation("/out.mp4", ran=ran)
    test_state.generation.release_generation_start_reservation()
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {}

    assert test_state.generation.try_reserve_generation_start()
    test_state.generation.start_generation("failed")
    assert run_peak._watch.active is True
    test_state.generation.fail_generation("boom")
    assert run_peak._watch.active is False
    test_state.generation.release_generation_start_reservation()
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {}

    assert test_state.generation.try_reserve_generation_start()
    test_state.generation.start_generation("finished")
    assert run_peak._watch.active is True
    test_state.generation.complete_generation("/out.mp4", ran=ran)
    assert run_peak._watch.active is False
    test_state.generation.release_generation_start_reservation()
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {
        _BUCKET: (FrontierPoint(point=ran.point, peak_growth_bytes=int(19.5 * _GIB)),)
    }


class _FailingCalibrationStore:
    def load_device_frontier(
        self, identity: DeviceIdentity, epoch: int
    ) -> Mapping[CalibrationBucket, tuple[FrontierPoint, ...]]:
        del identity, epoch
        return {}

    def load_device_machine_factor(
        self, identity: DeviceIdentity, epoch: int
    ) -> float | None:
        del identity, epoch
        return None

    def widen_device_frontier(
        self,
        identity: DeviceIdentity,
        epoch: int,
        bucket: CalibrationBucket,
        point: FrontierPoint,
        *,
        priced_gib: float,
        generation_id: str | None,
        conv_vae: bool = False,
    ) -> tuple[FrontierPoint, ...] | None:
        del identity, epoch, bucket, point, priced_gib, generation_id, conv_vae
        raise UnavailableError("database is locked")


def test_calibration_store_error_does_not_fail_generation(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    _configure_5090(test_state, fake_services, create_fake_model_files)
    test_state.calibration.bind(_FailingCalibrationStore())

    test_state.state.app_settings.budget_mode = "brave"
    response = client.post("/api/generate", json=_UNMEASURED_1440P)

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert _post_unmeasured(client, test_state, "cautious") == 422


def test_extrapolated_load_inside_recorded_envelope_admitted_in_cautious(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    _configure_5090(test_state, fake_services, create_fake_model_files)
    test_state.state.app_settings.budget_mode = "cautious"
    refused = client.post("/api/generate", json=_UNMEASURED_1440P)
    assert refused.status_code == 422
    assert refused.json()["code"] == VIDEO_JOB_TOO_LARGE

    assert _post_unmeasured(client, test_state, "brave") == 200

    assert _post_unmeasured(client, test_state, "cautious") == 200


def test_chunked_run_does_not_graduate_unchunked_job() -> None:
    chunked = JobShape(
        feature="fast",
        model_id="ltx-2.5-22b-distilled",
        adapter_id="",
        stage_mode="",
        tiled=False,
        chunked=True,
        seq=30_000,
        canvas_px=2560 * 1472,
        windows=3,
    )
    ran = envelope_for(chunked, "streaming_models_loading", 20.0)
    proof = FrontierPoint(point=ran.point, peak_growth_bytes=18 * _GIB)
    calibration = DeviceCalibration(identity=_5090, frontier={ran.bucket: (proof,)})
    unchunked = envelope_for(
        dataclasses.replace(chunked, chunked=False), "streaming_models_loading", 27.0
    )

    assert calibration.covering(unchunked) is None
    assert calibration.covering(ran) == proof


def test_old_epoch_rows_do_not_graduate(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    ran = _point(seq=60_000, canvas_px=2560 * 1472)
    _widen(store, ran)

    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH + 1) == {}
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {_BUCKET: (ran,)}


def test_frontier_reloaded_on_boot(
    client, test_state, fake_services, create_fake_model_files, default_app_settings
) -> None:
    _configure_5090(test_state, fake_services, create_fake_model_files)
    assert _post_unmeasured(client, test_state, "brave") == 200

    test_state.state.device_calibration = EMPTY_CALIBRATION
    test_state.load_persistent_state(default_app_settings)
    test_state.state.app_settings.use_local_text_encoder = True

    assert _post_unmeasured(client, test_state, "cautious") == 200


def test_device_calibration_frontier_matches_and_coalesces_rows_within_512_mib_total_memory_drift(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path)
    pre_reboot = _5090
    post_reboot = dataclasses.replace(_5090, total_memory_bytes=_5090.total_memory_bytes - 256 * _MIB)
    different_sku = dataclasses.replace(_5090, total_memory_bytes=16 * _GIB)
    small = _point(seq=30_000, canvas_px=1920 * 1088)
    dominating = _point(seq=60_000, canvas_px=2560 * 1472)

    store.widen_device_frontier(
        pre_reboot, CALIBRATION_EPOCH, _BUCKET, small, priced_gib=20.0, generation_id="g1"
    )
    assert store.load_device_frontier(post_reboot, CALIBRATION_EPOCH) == {_BUCKET: (small,)}
    assert store.load_device_frontier(different_sku, CALIBRATION_EPOCH) == {}

    widened = store.widen_device_frontier(
        post_reboot, CALIBRATION_EPOCH, _BUCKET, dominating, priced_gib=27.0, generation_id="g2"
    )
    assert widened == (dominating,)
    assert store.load_device_frontier(pre_reboot, CALIBRATION_EPOCH) == {_BUCKET: (dominating,)}
    assert store.load_device_frontier(post_reboot, CALIBRATION_EPOCH) == {_BUCKET: (dominating,)}


def test_clean_run_and_history_calibrate_device_machine_factor(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    timed = FrontierPoint(
        point=EnvelopePoint(seq=9216, canvas_px=1024 * 576, windows=1),
        peak_growth_bytes=8 * _GIB,
        wall_s=240.0,
    )
    store.widen_device_frontier(
        _5090, CALIBRATION_EPOCH, _BUCKET, timed, priced_gib=10.0, generation_id="g-timed"
    )
    factor = store.load_device_machine_factor(_5090, CALIBRATION_EPOCH)
    assert factor is not None and factor > 1.5




def test_smaller_run_does_not_rewrite_dominating_point_wall_time(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    big = FrontierPoint(
        point=EnvelopePoint(seq=9216, canvas_px=1024 * 576, windows=1),
        peak_growth_bytes=8 * _GIB,
        wall_s=312.0,
    )
    small = FrontierPoint(
        point=EnvelopePoint(seq=1440, canvas_px=512 * 288, windows=1),
        peak_growth_bytes=4 * _GIB,
        wall_s=229.0,
    )
    store.widen_device_frontier(
        _5090, CALIBRATION_EPOCH, _BUCKET, big, priced_gib=10.0, generation_id="g-big"
    )
    before = store.load_device_machine_factor(_5090, CALIBRATION_EPOCH)
    store.widen_device_frontier(
        _5090, CALIBRATION_EPOCH, _BUCKET, small, priced_gib=10.0, generation_id="g-small"
    )
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {_BUCKET: (big,)}
    assert store.load_device_machine_factor(_5090, CALIBRATION_EPOCH) == before

    rerun = dataclasses.replace(big, wall_s=300.0)
    store.widen_device_frontier(
        _5090, CALIBRATION_EPOCH, _BUCKET, rerun, priced_gib=10.0, generation_id="g-rerun"
    )
    assert store.load_device_frontier(_5090, CALIBRATION_EPOCH) == {_BUCKET: (rerun,)}


def test_frontier_inverts_wall_time_with_the_decode_the_run_used(tmp_path: Path) -> None:
    timed = FrontierPoint(
        point=EnvelopePoint(seq=17280, canvas_px=1280 * 704, windows=1),
        peak_growth_bytes=8 * _GIB,
        wall_s=314.0,
    )
    factors: dict[bool, float | None] = {}
    for conv_vae in (False, True):
        store = SqliteStore(tmp_path / str(conv_vae))
        store.widen_device_frontier(
            _5090,
            CALIBRATION_EPOCH,
            _BUCKET,
            timed,
            priced_gib=10.0,
            generation_id="g",
            conv_vae=conv_vae,
        )
        factors[conv_vae] = store.load_device_machine_factor(_5090, CALIBRATION_EPOCH)
    diff_vae, conv = factors[False], factors[True]
    # ConvVAE decodes faster, so more of the same wall time is streaming.
    assert diff_vae is not None and conv is not None and conv > diff_vae


def test_fresh_store_creates_the_frontier_with_timing_and_tile_columns(tmp_path: Path) -> None:
    SqliteStore(tmp_path)
    conn = sqlite3.connect(tmp_path / "store.sqlite3")
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(device_calibration_frontier)")}
    conn.close()
    assert {"wall_s", "conv_vae", "tiles"} <= columns


def test_tiled_run_is_inverted_with_the_tile_count_it_ran_with(tmp_path: Path) -> None:
    """The stored row keeps its tile count, so the factor does not absorb the tiles."""
    tiled_bucket = CalibrationBucket(
        model_id="ltx-2.5-22b-distilled",
        feature="iclora",
        adapter_id="restore",
        stage_mode="single_stage",
        load_mode="streaming_models_loading",
        tiled=True,
        chunked=False,
    )
    ran = EnvelopePoint(seq=9216, canvas_px=1280 * 704 * 2, windows=1, tiles=4)
    wall = estimate_envelope_wall_time(
        JobEnvelope(bucket=tiled_bucket, point=ran, priced_gib=10.0),
        system="Windows",
        machine_factor=1.0,
    )
    assert wall is not None
    store = SqliteStore(tmp_path)
    store.widen_device_frontier(
        _5090,
        CALIBRATION_EPOCH,
        tiled_bucket,
        FrontierPoint(point=ran, peak_growth_bytes=8 * _GIB, wall_s=float(wall.seconds)),
        priced_gib=10.0,
        generation_id="g-tiled",
    )

    stored = store.load_device_frontier(_5090, CALIBRATION_EPOCH)[tiled_bucket]
    assert [point.point.tiles for point in stored] == [4]
    factor = store.load_device_machine_factor(_5090, CALIBRATION_EPOCH)
    assert factor is not None and factor == pytest.approx(1.0, abs=0.05)


def test_history_bootstrap_ignores_api_models_and_fast_lora_recipes_as_ic_lora(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path)
    conn = sqlite3.connect(tmp_path / "store.sqlite3")
    t0 = 1_794_000_000_000
    meta = '{"mediaType":"video","metadata":{"width":1024,"height":576,"durationMs":5000,"fps":24}}'
    rows = [
        ("api-1", "text-to-video", '{"params":{"model":"fast-2.5","resolution":"540p"}}', 45_000),
        ("local-1", "claymation", '{"params":{"model":"ltx-2.5-fast","resolution":"540p"}}', 240_000),
        ("local-2", "text-to-video", '{"params":{"model":"ltx-2.5-fast","resolution":"540p"}}', 240_000),
        ("local-3", "jib-up", '{"params":{"model":"ltx-2.5-fast","resolution":"540p"}}', 240_000),
    ]
    for idx, (gid, feature, spec, dur_ms) in enumerate(rows):
        conn.execute(
            """
            INSERT INTO generations (
                id, feature, status, lane, contract_version, spec,
                created_at, queued_at, started_at, finished_at
            ) VALUES (?, ?, 'succeeded', 'local', 1, ?, ?, ?, ?, ?)
            """,
            (gid, feature, spec, t0 + idx, t0 + idx, t0, t0 + dur_ms),
        )
        conn.execute(
            """
            INSERT INTO assets (
                id, media_kind, origin, producer_generation_id, output_ordinal,
                mime_type, name, metadata, created_at
            ) VALUES (?, 'video', 'generated', ?, 0, 'video/mp4', ?, ?, ?)
            """,
            (f"asset-{gid}", gid, f"{gid}.mp4", meta, t0 + idx),
        )
    conn.commit()
    conn.close()

    factor = store.load_device_machine_factor(_5090, CALIBRATION_EPOCH)
    assert factor is not None and 2.5 < factor < 3.5


def _insert_history_run(
    conn: sqlite3.Connection, gid: str, feature: str, meta: str, *, dur_ms: int, idx: int
) -> None:
    t0 = 1_794_000_000_000
    conn.execute(
        """
        INSERT INTO generations (
            id, feature, status, lane, contract_version, spec,
            created_at, queued_at, started_at, finished_at
        ) VALUES (?, ?, 'succeeded', 'local', 1, ?, ?, ?, ?, ?)
        """,
        (
            gid,
            feature,
            '{"params":{"model":"ltx-2.5-fast","resolution":"540p"}}',
            t0 + idx,
            t0 + idx,
            t0,
            t0 + dur_ms,
        ),
    )
    conn.execute(
        """
        INSERT INTO assets (
            id, media_kind, origin, producer_generation_id, output_ordinal,
            mime_type, name, metadata, created_at
        ) VALUES (?, 'video', 'generated', ?, 0, 'video/mp4', ?, ?, ?)
        """,
        (f"asset-{gid}", gid, f"{gid}.mp4", meta, t0 + idx),
    )


def test_history_skips_rows_with_odd_stored_values_instead_of_raising(tmp_path: Path) -> None:
    """A float or an absurd duration in stored metadata must not abort the startup load."""
    store = SqliteStore(tmp_path)
    conn = sqlite3.connect(tmp_path / "store.sqlite3")
    good = '{"mediaType":"video","metadata":{"width":1024,"height":576,"durationMs":5000,"fps":24}}'
    float_ms = '{"mediaType":"video","metadata":{"width":1024,"height":576,"durationMs":5000.5,"fps":24}}'
    huge_ms = '{"mediaType":"video","metadata":{"width":1024,"height":576,"durationMs":1000000000000,"fps":24}}'
    runs = [(good, 240_000)] * 3 + [(float_ms, 240_000), (huge_ms, 240_000)]
    for idx, (meta, dur_ms) in enumerate(runs):
        _insert_history_run(conn, f"run-{idx}", "text-to-video", meta, dur_ms=dur_ms, idx=idx)
    conn.commit()
    conn.close()

    factor = store.load_device_machine_factor(_5090, CALIBRATION_EPOCH)

    # The three good rows still calibrate. The odd rows are skipped.
    assert factor is not None and 2.5 < factor < 3.5


def test_same_shape_rerun_takes_over_the_frontier_row_generation_id(tmp_path: Path) -> None:
    """The refreshed row belongs to the new run, so the history query does not count it twice."""
    store = SqliteStore(tmp_path)
    point = FrontierPoint(
        point=EnvelopePoint(seq=9216, canvas_px=1024 * 576, windows=1),
        peak_growth_bytes=8 * _GIB,
        wall_s=312.0,
    )
    store.widen_device_frontier(
        _5090, CALIBRATION_EPOCH, _BUCKET, point, priced_gib=10.0, generation_id="g-first"
    )
    store.widen_device_frontier(
        _5090,
        CALIBRATION_EPOCH,
        _BUCKET,
        dataclasses.replace(point, wall_s=300.0),
        priced_gib=10.0,
        generation_id="g-rerun",
    )

    conn = sqlite3.connect(tmp_path / "store.sqlite3")
    rows = conn.execute("SELECT generation_id, wall_s FROM device_calibration_frontier").fetchall()
    conn.close()

    assert rows == [("g-rerun", 300.0)]


class _RefreshFailsStore(SqliteStore):
    def load_device_machine_factor(self, identity, epoch):  # type: ignore[no-untyped-def]
        raise UnavailableError()


def test_failed_machine_factor_refresh_keeps_the_recorded_frontier_point(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    _configure_5090(test_state, fake_services, create_fake_model_files)
    handler = test_state.calibration
    handler.bind(_RefreshFailsStore(tmp_path))
    ran = _job(9216, 1024 * 576)

    verdict = handler.record_if_clean(ran, _START, _end(peak_gib=20.0), generation_id="g", wall_s=240.0)

    assert verdict == "clean"
    recorded = test_state.state.device_calibration.frontier[_BUCKET]
    assert [point.point for point in recorded] == [ran.point]
    assert test_state.state.device_calibration.machine_factor is None

