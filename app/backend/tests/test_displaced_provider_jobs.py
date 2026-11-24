"""The file that keeps the provider job ids that a retry replaced."""

from __future__ import annotations

import json
from pathlib import Path

from services.displaced_provider_jobs import (
    DISPLACED_PROVIDER_JOBS_FILENAME,
    DisplacedProviderJobs,
)
from services.records import ProviderOrphan


def _job(generation_id: str = "gen-1", provider_job_id: str = "req-1") -> ProviderOrphan:
    return ProviderOrphan(generation_id=generation_id, lane="fal", provider_job_id=provider_job_id)


def _file(tmp_path: Path) -> Path:
    return tmp_path / DISPLACED_PROVIDER_JOBS_FILENAME


def test_a_new_file_is_empty_and_is_not_created_by_a_read(tmp_path: Path) -> None:
    jobs = DisplacedProviderJobs(_file(tmp_path))

    assert jobs.list() == []
    assert not _file(tmp_path).exists()


def test_an_added_id_is_kept_across_instances_in_the_order_it_was_added(tmp_path: Path) -> None:
    # Bug caught: a retry keeps an id in memory only, and a restart loses the handle to a paid job.
    DisplacedProviderJobs(_file(tmp_path)).add(_job("gen-2", "req-b"))
    DisplacedProviderJobs(_file(tmp_path)).add(_job("gen-1", "req-a"))

    kept = DisplacedProviderJobs(_file(tmp_path)).list()

    assert [(job.generation_id, job.provider_job_id) for job in kept] == [
        ("gen-2", "req-b"),
        ("gen-1", "req-a"),
    ]
    assert [job.provider_job_id for job in DisplacedProviderJobs(_file(tmp_path)).list("gen-1")] == ["req-a"]


def test_adding_the_same_id_twice_keeps_one_entry(tmp_path: Path) -> None:
    jobs = DisplacedProviderJobs(_file(tmp_path))

    jobs.add(_job())
    jobs.add(_job())

    assert len(jobs.list()) == 1


def test_a_removed_id_is_gone_and_the_file_is_deleted_with_the_last_entry(tmp_path: Path) -> None:
    # Bug caught: the file stays on disk for ever, and the boot cancels the same jobs again.
    jobs = DisplacedProviderJobs(_file(tmp_path))
    jobs.add(_job(provider_job_id="req-1"))
    jobs.add(_job(provider_job_id="req-2"))

    assert jobs.remove("gen-1", "req-1") is True
    assert [job.provider_job_id for job in jobs.list()] == ["req-2"]
    assert _file(tmp_path).exists()
    assert jobs.remove("gen-1", "req-1") is False

    assert jobs.remove("gen-1", "req-2") is True
    assert not _file(tmp_path).exists()


def test_an_unreadable_file_is_moved_aside_and_the_list_is_empty(tmp_path: Path) -> None:
    # Bug caught: a half-written file stops the app at every boot.
    _file(tmp_path).write_text("{not json", encoding="utf-8")
    jobs = DisplacedProviderJobs(_file(tmp_path))

    assert jobs.list() == []

    assert not _file(tmp_path).exists()
    assert (tmp_path / f"{DISPLACED_PROVIDER_JOBS_FILENAME}.corrupt").read_text(encoding="utf-8") == "{not json"
    jobs.add(_job())
    assert len(jobs.list()) == 1


def test_an_invalid_entry_is_skipped_and_the_valid_ones_are_kept(tmp_path: Path) -> None:
    good = _job().model_dump(mode="json")
    _file(tmp_path).write_text(
        json.dumps({"version": 1, "jobs": [{"generation_id": "gen-1"}, good, {**good, "provider_job_id": ""}]}),
        encoding="utf-8",
    )

    assert [job.provider_job_id for job in DisplacedProviderJobs(_file(tmp_path)).list()] == ["req-1"]
