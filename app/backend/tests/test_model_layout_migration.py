"""Tests for legacy model layout migration."""

from __future__ import annotations

from server_utils.model_layout_migration import migrate_legacy_models_layout


def test_migration_moves_legacy_contents_and_cleans_empty_legacy_dir(tmp_path):
    app_data = tmp_path / "app_data"
    legacy_root = app_data / "models" / "ltx-2"
    legacy_root.mkdir(parents=True, exist_ok=True)

    checkpoint_legacy = legacy_root / "ltx-2-19b-distilled-fp8.safetensors"
    checkpoint_legacy.write_bytes(b"legacy-checkpoint")
    text_encoder_legacy = legacy_root / "text_encoder"
    text_encoder_legacy.mkdir(parents=True, exist_ok=True)
    (text_encoder_legacy / "model.safetensors").write_bytes(b"legacy-te")

    migrate_legacy_models_layout(app_data)

    models_root = app_data / "models"
    assert (models_root / "ltx-2-19b-distilled-fp8.safetensors").read_bytes() == b"legacy-checkpoint"
    assert (models_root / "text_encoder" / "model.safetensors").read_bytes() == b"legacy-te"
    assert not legacy_root.exists()


def test_migration_keeps_existing_targets_and_leaves_conflicting_legacy_items(tmp_path):
    app_data = tmp_path / "app_data"
    models_root = app_data / "models"
    legacy_root = models_root / "ltx-2"
    legacy_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    target_checkpoint = models_root / "ltx-2-19b-distilled-fp8.safetensors"
    target_checkpoint.write_bytes(b"new-target")
    legacy_checkpoint = legacy_root / "ltx-2-19b-distilled-fp8.safetensors"
    legacy_checkpoint.write_bytes(b"old-legacy")

    migrate_legacy_models_layout(app_data)

    assert target_checkpoint.read_bytes() == b"new-target"
    assert legacy_checkpoint.read_bytes() == b"old-legacy"
    assert legacy_root.exists()


def test_migration_removes_empty_legacy_dir(tmp_path):
    app_data = tmp_path / "app_data"
    legacy_root = app_data / "models" / "ltx-2"
    legacy_root.mkdir(parents=True, exist_ok=True)

    migrate_legacy_models_layout(app_data)

    assert not legacy_root.exists()
