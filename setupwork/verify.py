"""Validate the native NSIS records and the ZIP overlay from the final executable."""
import hashlib
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_installer(exe, stub, sevenzip, embedded, index):
    # NSIS sees only the bounded stub. It must have its own CRC even with a ZIP appended.
    head = stub.read_bytes()
    pe, = struct.unpack_from("<I", head, 0x3C)
    if head[:2] != b"MZ" or head[pe:pe + 6] != b"PE\0\0\x64\x86":
        raise ValueError("Expected an AMD64 Windows installer")
    tag = head.index(b"\xef\xbe\xad\xdeNullsoftInst") - 4
    extent, = struct.unpack_from("<I", head, tag + 24)
    crc, = struct.unpack_from("<I", head, len(head) - 4)
    if tag + extent != len(head) or zlib.crc32(head[512:-4]) != crc:
        raise ValueError("Invalid NSIS size or CRC")
    with exe.open("rb") as stream:
        if stream.read(len(head)) != head or stream.read(4) != b"PK\x03\x04":
            raise ValueError("Installer code or appended ZIP differs from the build")

    # Some 7-Zip releases can list the x64 NSIS table but cannot extract it.
    # Read the copy/zlib records directly and verify the contents by filename.
    listing = subprocess.check_output([sevenzip, "l", "-tNsis", "-slt", str(stub)], text=True)
    packed_header, = struct.unpack_from("<I", head, tag + 28)
    record_base = tag + 32 + (packed_header & 0x7fffffff)
    found = set()
    for record in listing.split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in record.splitlines() if " = " in line)
        name = fields.get("Path", "").replace("\\", "/")
        if not name.startswith("$PLUGINSDIR/bundle/"):
            continue
        name = name.removeprefix("$PLUGINSDIR/bundle/")
        start = record_base + int(fields["Offset"])
        length, = struct.unpack_from("<I", head, start)
        data = head[start + 4:start + 4 + (length & 0x7fffffff)]
        if length & 0x80000000:
            data = zlib.decompress(data, -15)
        if name in found or name not in embedded or hashlib.sha256(data).hexdigest() != sha256(embedded[name]):
            raise ValueError(f"Unexpected or damaged NSIS payload: {name}")
        found.add(name)
    if found != embedded.keys():
        raise ValueError(f"Missing NSIS payloads: {sorted(embedded.keys() - found)}")

    with zipfile.ZipFile(exe) as media:
        if sorted(media.namelist()) != sorted(index):
            raise ValueError("Unexpected files in installer overlay")
        for name, item in index.items():
            entry = media.getinfo(name)
            if entry.file_size != item["size"] or entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Wrong overlay size or method: {name}")
            with media.open(entry) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError(f"Wrong overlay digest: {name}")
