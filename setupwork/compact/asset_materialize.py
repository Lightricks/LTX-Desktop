"""Recreate the release's app resources from verified blobs. Uses only Python's standard library."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from zipfile import ZipFile

CHUNK = 1024 * 1024


def _file_matches(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def _destination(root, name):
    path = PureWindowsPath(name)
    if not name or path.anchor or "\\" in name or any(
        piece in ("", ".", "..") or ":" in piece for piece in name.split("/")
    ):
        raise ValueError(f"Invalid resource destination: {name!r}")
    return root.joinpath(*name.split("/"))


@contextmanager
def _atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False, prefix=".ltx-") as output:
        temporary = Path(output.name)
        try:
            yield output
            output.close()
            os.replace(temporary, path)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def _raw_record(recipe, digest, size, name, lfs):
    root = "https://media.githubusercontent.com/media" if lfs else "https://raw.githubusercontent.com"
    return {
        "filename": f"ltx-asset-{digest}.bin",
        "url": f"{root}/{recipe['repo']}/{recipe['revision']}/{quote(name, safe='/')}",
        "size": size,
        "sha256": digest,
    }


def hydrate(stage: Path, resources: Path, cache: Path, fetch):
    """Write stage/resources. fetch(record, cache) returns a verified local download Path."""
    target = Path(stage) / "resources"
    with ZipFile(Path(resources) / "asset-payload.zip") as bundle, ExitStack() as opened:
        recipe = json.loads(bundle.read("recipe.json"))
        if recipe["format"] != 1:
            raise ValueError("Unsupported application asset recipe.")
        archive = recipe["asar"]
        pending_asar = not _file_matches(target / "app.asar", archive["size"], archive["sha256"])
        pending = []
        for name, digest in recipe["files"]:
            dest = _destination(target, name)
            if not _file_matches(dest, recipe["blobs"][digest][0], digest):
                pending.append((dest, digest))
        if not pending_asar and not pending:
            return

        upstream = None
        needed = {h for _, h in pending}
        if pending_asar:
            needed.update(archive["parts"])
        if any(recipe["blobs"][h][1] == "source" for h in needed):
            try:
                upstream = opened.enter_context(ZipFile(fetch(recipe["upstream"], cache)))
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"Source archive unavailable ({type(exc).__name__}); downloading individual verified files.", flush=True)

        def copy_blob(output, digest, aggregate=None):
            size, kind, name = recipe["blobs"][digest]
            if kind == "source" and upstream is not None:
                data = upstream.open(recipe["prefix"] + name)
            elif kind in ("source", "lfs"):
                record = _raw_record(recipe, digest, size, name, lfs=(kind == "lfs"))
                data = Path(fetch(record, cache)).open("rb")
            elif kind == "embedded":
                data = bundle.open("blobs/" + digest)
            else:
                raise ValueError(f"Unknown application asset source: {kind}")
            count = 0
            actual = hashlib.sha256()
            with data:
                while chunk := data.read(CHUNK):
                    count += len(chunk)
                    if count > size:
                        raise ValueError(f"Application asset is oversized: {name or digest}")
                    actual.update(chunk)
                    if aggregate is not None:
                        aggregate.update(chunk)
                    output.write(chunk)
            if count != size or actual.hexdigest() != digest:
                raise ValueError(f"Application asset failed verification: {name or digest}")

        if pending_asar:
            with _atomic_file(target / "app.asar") as out:
                full_hash = hashlib.sha256()
                for digest in archive["parts"]:
                    copy_blob(out, digest, full_hash)
                if out.tell() != archive["size"] or full_hash.hexdigest() != archive["sha256"]:
                    raise ValueError("Reconstructed app.asar failed verification.")
        for path, digest in pending:
            with _atomic_file(path) as out:
                copy_blob(out, digest)
