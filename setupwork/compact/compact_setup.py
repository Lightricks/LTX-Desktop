"""Prepare the same desktop tree as the full AMD online installer, from smaller media."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import asset_materialize
import vendor_materialize


def inventory(resources: Path) -> dict:
    return json.loads((resources / "shell-inventory.json").read_text(encoding="utf-8"))


def required_bytes(resources: Path) -> int:
    manifest = inventory(resources)
    return manifest["downloadBytes"] + sum(size for size, _ in manifest["files"].values())


def verify_tree(stage: Path, manifest: dict) -> None:
    files = manifest["files"]
    found = {
        candidate.relative_to(stage).as_posix()
        for candidate in stage.rglob("*") if candidate.is_file()
    }
    found.discard(".ltx-amd-managed-v1")
    if found != set(files):
        missing = sorted(set(files) - found)
        extra = sorted(found - set(files))
        raise ValueError(f"Desktop inventory differs: missing {missing[:2]}, unexpected {extra[:2]}.")
    for name, (size, digest) in files.items():
        path = stage / name
        if path.stat().st_size == size:
            with path.open("rb") as data:
                if hashlib.file_digest(data, "sha256").hexdigest() == digest:
                    continue
        raise ValueError(f"Desktop file failed verification: {name}")


def hydrate(stage, resources, cache, download_file):
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = set()

    def fetch(record, directory):
        name = record["filename"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid component filename")
        # Names come from the embedded release manifests. The common downloader
        # checks the expected size and digest for both saved and new downloads.
        target = directory / name
        logging.info("Fetching or verifying %s (%.1f MB)", name, record["size"] / 1e6)
        download_file(record, target)
        downloaded.add(target)
        return target

    logging.info("Preparing the Electron desktop and the private video engine.")
    vendor_materialize.hydrate(stage, resources, cache, fetch)
    logging.info("Preparing LTX Desktop resources.")
    asset_materialize.hydrate(stage, resources, cache, fetch)
    logging.info("Checking the installed desktop files against the original release.")
    verify_tree(stage, inventory(resources))
    return downloaded


def clear_downloads(paths):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logging.warning("Keeping cached component %s: %s", path.name, exc)
