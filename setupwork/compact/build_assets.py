"""Build the compact asset recipe from an already verified staged Windows shell and its pinned upstream ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

REPO = "Lightricks/LTX-Desktop"
REV = "2ab055b98f7590581c56f9e0b2ff55ac56d459b7"
PREFIX = "LTX-Desktop-" + REV + "/"
UPSTREAM = {
    "filename": "ltx-desktop-upstream-" + REV + ".zip",
    "size": 198_280_031,
    "sha256": "d3c56288c3db59d8dd4202c377e27473364c82dcc61a7805ca1190e84e42349b",
    "url": f"https://codeload.github.com/{REPO}/zip/{REV}",
}
LFS = re.compile(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize ([0-9]+)\n?")
EMBED_BELOW = 32 * 1024


def source_members(filename):
    result = {}
    with filename.open("rb") as raw:
        if filename.stat().st_size != UPSTREAM["size"] or hashlib.file_digest(raw, "sha256").hexdigest() != UPSTREAM["sha256"]:
            raise ValueError("The upstream source ZIP does not match the pinned release.")
    with ZipFile(filename) as archive:
        for item in archive.infolist():
            if item.is_dir() or not item.filename.startswith(PREFIX):
                continue
            name = item.filename[len(PREFIX):]
            if item.file_size >= EMBED_BELOW:
                with archive.open(item) as data:
                    digest = hashlib.file_digest(data, "sha256").hexdigest()
                result[digest] = (item.file_size, "source", name)
            elif 125 <= item.file_size <= 1024:
                match = LFS.fullmatch(archive.read(item).replace(b"\r\n", b"\n"))
                if match:
                    digest, size = match.groups()
                    result.setdefault(digest.decode(), (int(size), "lfs", name))
    return result


def add_entry(bundle, name, data):
    # Fixed metadata allows identical builds on Windows and Unix.
    info = ZipInfo(name, (2020, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)


def asar_spans(data):
    """Preserve Chromium pickle header, exact packed file boundaries, and every padding byte."""
    word_size, payload_size, pickle_size, json_size = struct.unpack_from("<4I", data)
    base = 8 + payload_size
    if word_size != 4 or pickle_size + 4 != payload_size or not 16 + json_size <= base <= len(data):
        raise ValueError("Invalid Electron ASAR header.")
    entries = []
    def walk(files):
        for v in files.values():
            if "files" in v:
                walk(v["files"])
            elif "offset" in v and not v.get("unpacked") and v["size"]:
                start = base + int(v["offset"])
                entries.append((start, start + v["size"]))
    walk(json.loads(data[16:16+json_size])["files"])
    yield data[:base]
    cursor = base
    for start, end in sorted(entries):
        if start < cursor or end < start or end > len(data):
            raise ValueError("Invalid ASAR entry boundaries.")
        if start > cursor:
            yield data[cursor:start]
        yield data[start:end]
        cursor = end
    if cursor < len(data):
        yield data[cursor:]


def build(shell, source, target):
    sources = source_members(source)
    resources = shell / "resources"
    target.parent.mkdir(parents=True, exist_ok=True)
    blobs = {}
    with ZipFile(target, "w") as bundle:
        def add(data):
            digest = hashlib.sha256(data).hexdigest()
            if digest not in blobs:
                if digest in sources:
                    if sources[digest][0] != len(data):
                        raise ValueError("Conflicting blob sizes.")
                    blobs[digest] = sources[digest]
                else:
                    blobs[digest] = (len(data), "embedded", "")
                    add_entry(bundle, "blobs/" + digest, data)
            return digest
        asar = (resources / "app.asar").read_bytes()
        archive = {"size": len(asar), "sha256": hashlib.sha256(asar).hexdigest(),
                   "parts": [add(span) for span in asar_spans(asar)]}
        del asar
        files = []
        for path in sorted(resources.rglob("*")):
            relative = path.relative_to(resources)
            if relative.parts[0] in ("comfy-engine", "app.asar") or not path.is_file():
                continue
            files.append((relative.as_posix(), add(path.read_bytes())))
        recipe = {"format": 1, "repo": REPO, "revision": REV, "prefix": PREFIX,
                  "upstream": UPSTREAM, "asar": archive, "files": files, "blobs": blobs}
        add_entry(bundle, "recipe.json", json.dumps(recipe, ensure_ascii=False, separators=(",", ":")).encode())
    print(f"{target.stat().st_size:,}-byte payload; {len(archive['parts'])} ASAR spans, {len(files)} loose files, "
          f"{sum(r[1] != 'embedded' for r in blobs.values())} unique upstream assets.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shell", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "payload" / "asset-payload.zip")
    a = p.parse_args()
    build(a.shell, a.source, a.output)
