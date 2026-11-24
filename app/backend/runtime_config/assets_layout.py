"""Where the asset store keeps its files. One definition, used by the store and its callers."""

from __future__ import annotations

from pathlib import Path

IN_FLIGHT_DIRNAME = ".in-flight"


def assets_folder(app_data_dir: Path, assets_dir: Path | None = None) -> Path:
    """The user-chosen assets folder (LTX_ASSETS_DIR), or ``<app_data_dir>/assets``."""
    return assets_dir if assets_dir is not None else app_data_dir / "assets"


def in_flight_folder(app_data_dir: Path, assets_dir: Path | None = None) -> Path:
    return assets_folder(app_data_dir, assets_dir) / IN_FLIGHT_DIRNAME
