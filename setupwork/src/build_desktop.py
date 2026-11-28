"""Archive the shell, recording ASAR slices for identical standalone assets."""
import hashlib
import json
from pathlib import Path
import struct
import zipfile


def asar_offsets(node, base, groups):
    for item in node["files"].values():
        if "files" in item:
            asar_offsets(item, base, groups)
        elif "offset" in item and item["size"] > 65536:
            groups.setdefault(item["size"], []).append(base + int(item["offset"]))


def region_hash(blob, offset, length):
    blob.seek(offset)
    h = hashlib.sha256()
    while length:
        data = blob.read(min(length, 1024**2))
        if not data:
            raise ValueError("ASAR file is truncated")
        length -= len(data)
        h.update(data)
    return h.hexdigest()


def build(base, out):
    asar = base / "resources/app.asar"
    groups, hashed_offsets, aliases = {}, {}, []
    with asar.open("rb") as blob:
        _, header_length, _, json_size = struct.unpack("<IIII", blob.read(16))
        asar_offsets(json.loads(blob.read(json_size)), 8 + header_length, groups)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for file in sorted(base.rglob("*")):
                if file.is_symlink():
                    raise ValueError(f"Unexpected symlink in Windows shell: {file}")
                if not file.is_file():
                    continue
                relative = file.relative_to(base).as_posix()
                length = file.stat().st_size
                candidates = groups.get(length, []) if file != asar else []
                if candidates:
                    with file.open("rb") as asset:
                        digest = hashlib.file_digest(asset, "sha256").hexdigest()
                    matched = False
                    for offset in candidates:
                        if offset not in hashed_offsets:
                            hashed_offsets[offset] = region_hash(blob, offset, length)
                        if digest == hashed_offsets[offset]:
                            aliases.append({"path": relative, "offset": offset, "size": length, "sha256": digest})
                            matched = True
                            break
                    if matched:
                        continue
                # The bootstrap's bundled Python includes LZMA. The large Electron
                # and native renderer binaries compress substantially better with it.
                compression = zipfile.ZIP_DEFLATED
                if length >= 5 * 1024**2 and file.suffix.lower() in {".exe", ".dll", ".pyd", ".html"}:
                    compression = zipfile.ZIP_LZMA
                archive.write(file, relative, compress_type=compression)
            archive.writestr("amd-asset-offsets.json", json.dumps(aliases))
    print(f"Archived {Path(out).name} with {len(aliases)} ASAR asset references", flush=True)
