"""Direct tests for the filesystem-backed media-store lifecycle."""

from __future__ import annotations

import json
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import pytest

from services.media_store.filesystem_media_store import FilesystemMediaStore
from services.media_store.media_store import MediaTooLargeError


def _make_store(
    tmp_path: Path,
    *,
    upload_ttl: timedelta = timedelta(hours=24),
    artifact_ttl: timedelta = timedelta(days=7),
    cleanup_interval: timedelta = timedelta(0),
    delete_expired_artifact_files: bool = False,
) -> tuple[FilesystemMediaStore, Path, Path]:
    app_data_dir = tmp_path / "filesystem-store-app-data"
    outputs_dir = tmp_path / "filesystem-store-outputs"
    outputs_dir.mkdir()
    store = FilesystemMediaStore(
        app_data_dir=app_data_dir,
        outputs_dir=outputs_dir,
        upload_ttl=upload_ttl,
        artifact_ttl=artifact_ttl,
        cleanup_interval=cleanup_interval,
        delete_expired_artifact_files=delete_expired_artifact_files,
    )
    return store, app_data_dir, outputs_dir


def _artifact_metadata_path(app_data_dir: Path, artifact_id: str) -> Path:
    return app_data_dir / "media" / "artifacts" / artifact_id / "metadata.json"


def test_managed_local_expiry_preserves_output_file(tmp_path: Path) -> None:
    store, app_data_dir, outputs_dir = _make_store(
        tmp_path,
        artifact_ttl=timedelta(seconds=-1),
        delete_expired_artifact_files=False,
    )
    output = outputs_dir / "generated.mp4"
    output.write_bytes(b"video")
    artifact = store.register_artifact(output, media_type="video", content_type="video/mp4")

    store.cleanup_expired()

    assert output.is_file()
    assert not _artifact_metadata_path(app_data_dir, artifact.id).parent.exists()
    assert store.resolve_artifact(artifact.id) is None


def test_standalone_expiry_deletes_owned_output_file(tmp_path: Path) -> None:
    store, app_data_dir, outputs_dir = _make_store(
        tmp_path,
        artifact_ttl=timedelta(seconds=-1),
        delete_expired_artifact_files=True,
    )
    output = outputs_dir / "generated.mp4"
    output.write_bytes(b"video")
    artifact = store.register_artifact(output, media_type="video", content_type="video/mp4")

    store.cleanup_expired()

    assert not output.exists()
    assert not _artifact_metadata_path(app_data_dir, artifact.id).parent.exists()
    assert store.resolve_artifact(artifact.id) is None


def test_expired_upload_content_is_removed(tmp_path: Path) -> None:
    store, _app_data_dir, _outputs_dir = _make_store(
        tmp_path,
        upload_ttl=timedelta(seconds=-1),
    )
    staged = store.stage_upload(BytesIO(b"image"), filename="frame.png", max_bytes=100)
    uploaded = store.commit_upload(
        staged,
        media_type="image",
        filename="frame.png",
        content_type="image/png",
    )

    store.cleanup_expired()

    assert not uploaded.path.exists()
    assert store.resolve_upload(uploaded.id) is None


def test_cleanup_is_throttled_after_first_scan(tmp_path: Path) -> None:
    store, app_data_dir, outputs_dir = _make_store(
        tmp_path,
        artifact_ttl=timedelta(seconds=-1),
        cleanup_interval=timedelta(hours=1),
    )
    first_output = outputs_dir / "first.mp4"
    first_output.write_bytes(b"first")
    first = store.register_artifact(first_output, media_type="video", content_type="video/mp4")
    store.cleanup_expired()
    assert not _artifact_metadata_path(app_data_dir, first.id).parent.exists()

    second_output = outputs_dir / "second.mp4"
    second_output.write_bytes(b"second")
    second = store.register_artifact(second_output, media_type="video", content_type="video/mp4")
    store.cleanup_expired()

    assert _artifact_metadata_path(app_data_dir, second.id).is_file()


def test_oversize_upload_removes_staged_file(tmp_path: Path) -> None:
    store, app_data_dir, _outputs_dir = _make_store(tmp_path)

    with pytest.raises(MediaTooLargeError):
        store.stage_upload(BytesIO(b"too large"), filename="frame.png", max_bytes=3)

    assert list((app_data_dir / "media" / "staging").iterdir()) == []


def test_corrupt_artifact_metadata_is_rejected_without_deleting_output(tmp_path: Path) -> None:
    store, app_data_dir, outputs_dir = _make_store(tmp_path)
    output = outputs_dir / "generated.mp4"
    output.write_bytes(b"video")
    artifact = store.register_artifact(output, media_type="video", content_type="video/mp4")
    metadata_path = _artifact_metadata_path(app_data_dir, artifact.id)
    metadata_path.write_text("not-json", encoding="utf-8")

    assert store.resolve_artifact(artifact.id) is None
    store.cleanup_expired()
    assert output.is_file()


def test_expired_traversal_metadata_cannot_delete_outside_outputs(tmp_path: Path) -> None:
    store, app_data_dir, outputs_dir = _make_store(
        tmp_path,
        artifact_ttl=timedelta(seconds=-1),
        delete_expired_artifact_files=True,
    )
    output = outputs_dir / "generated.mp4"
    output.write_bytes(b"video")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    artifact = store.register_artifact(output, media_type="video", content_type="video/mp4")
    metadata_path = _artifact_metadata_path(app_data_dir, artifact.id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["relative_path"] = "../outside.mp4"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    store.cleanup_expired()

    assert outside.read_bytes() == b"outside"
    assert output.read_bytes() == b"video"
    assert not metadata_path.parent.exists()
