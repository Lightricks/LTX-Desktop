#!/usr/bin/env python3
"""Restore the compact checkout's original media and vendor archives. Requires Python 3.11+."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path, PureWindowsPath
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


def relative_name(name):
    if (not name or PureWindowsPath(name).anchor or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Invalid relative path: {name!r}")
    return name


def destination(root, name):
    path = root / relative_name(name)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Refusing to follow a link outside the checkout or replace a link: {name}")
    return path


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def fetch(url):
    request = Request(url, headers={"User-Agent": "LTX-Desktop-AMD-source-restore"})
    return urlopen(request, timeout=90)


def verified_copy(source, target, record, *, cache=False):
    """Stage and verify bytes next to their destination before making them visible."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ltx-", delete=False) as output:
        temp = Path(output.name)
    try:
        with temp.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if not matches(temp, record):
            raise ValueError(f"Size or SHA-256 mismatch for {target.name}")
        if not cache and (target.exists() or target.is_symlink()):
            if not target.is_symlink() and matches(target, record):
                return
            raise ValueError(f"Existing file differs; leaving it untouched: {target}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def desktop_zip(record, override, cache_dir):
    if override is not None:
        # Individual members are still size- and hash-verified on extraction.
        return override
    archive = destination(cache_dir, record["filename"])
    if not matches(archive, record):
        print(f"Downloading pinned Desktop sources to {archive}", flush=True)
        with fetch(record["url"]) as source:
            verified_copy(source, archive, record, cache=True)
    return archive


def run(args, root):
    manifest = json.loads((root / "source-assets.json").read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise ValueError("Unsupported source-assets.json version")
    missing = []
    changed = []
    for record in manifest["files"]:
        path = destination(root, record["path"])
        if path.exists():
            if not matches(path, record):
                changed.append(record["path"])
        else:
            missing.append((record, path))
    if changed:
        raise ValueError("Existing files differ. Move them aside to restore the originals:\n"
                         + "\n".join(changed))
    if args.verify_only and missing:
        raise ValueError(f"{len(missing)} files are missing. Run restore_sources.py to restore them.\n"
                         + "\n".join(record["path"] for record, _ in missing[:5]))
    cache_dir = (args.cache_dir or (root / "build/.cache/source-archives")).resolve()
    desktop = manifest["desktop"]
    with ExitStack() as stack:
        archive = None
        for record, target in missing:
            if "member" in record:
                if archive is None:
                    file = desktop_zip(desktop, args.desktop_zip, cache_dir)
                    archive = stack.enter_context(ZipFile(file))
                member = desktop["prefix"].rstrip("/") + "/" + relative_name(record["member"])
                source = archive.open(member)
            elif args.vendor_dir:
                source = (args.vendor_dir / target.name).open("rb")
            else:
                print(f"Downloading {target.name}", flush=True)
                source = fetch(record["url"])
            with source:
                verified_copy(source, target, record)
    print(f"Verified {len(manifest['files'])} source assets ({len(missing)} restored).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop-zip", type=Path,
                        help="Read Desktop media from a local source ZIP with the pinned archive's layout.")
    parser.add_argument("--vendor-dir", type=Path,
                        help="Read the named vendor ZIPs from this directory instead of downloading them.")
    parser.add_argument("--cache-dir", type=Path,
                        help="Directory for the pinned Desktop ZIP (default: build/.cache/source-archives).")
    parser.add_argument("--verify-only", action="store_true", help="Check all files without writing or downloading.")
    try:
        run(parser.parse_args(argv), Path(__file__).resolve().parent)
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        print(f"Source restore failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
