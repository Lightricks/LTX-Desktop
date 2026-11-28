"""Stage the published Kandinsky Registry pack in the private Comfy node directory."""

from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from pathlib import Path

# Published by kandinskylab at https://cdn.comfy.org/kandinskylab/kandinsky6/1.0.2/node.zip
ARCHIVE_NAME = "kandinsky6-1.0.2.zip"
ARCHIVE_BYTES = 2_204_011
ARCHIVE_SHA256 = "5ddf0bef14d0e571f8760403d270f2b1d6f9b6b10e56b8b707743c15751b7f80"
DIRECTORY = "kandinsky6"
_MARKER = ".ltx-registry-sha256"


class KandinskyPackError(RuntimeError):
    pass


def pack_installed(custom_nodes: Path) -> bool:
    target = custom_nodes / DIRECTORY
    if target.is_symlink():
        raise KandinskyPackError("The private Kandinsky node folder is a symbolic link. Move it aside and retry setup.")
    if not target.exists():
        return False
    marker = target / _MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == ARCHIVE_SHA256:
        return True
    raise KandinskyPackError(
        f"The private Kandinsky node folder already contains another installation. Move {target} aside, then retry setup."
    )


def install_pack(custom_nodes: Path, archive: Path) -> None:
    if pack_installed(custom_nodes):
        return
    if not archive.is_file():
        raise KandinskyPackError("The bundled Kandinsky node archive is missing. Reinstall this version of LTX Desktop - AMD 395 MAX.")
    with archive.open("rb") as source:
        payload = source.read(ARCHIVE_BYTES + 1)
    if len(payload) != ARCHIVE_BYTES or hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise KandinskyPackError("The bundled Kandinsky node archive failed its integrity check. Reinstall LTX Desktop - AMD 395 MAX.")
    custom_nodes.mkdir(parents=True, exist_ok=True)
    # Prepare the complete pinned archive outside the scanned directory, on the same volume.
    with tempfile.TemporaryDirectory(prefix=".ltx-kandinsky-", dir=custom_nodes.parent) as staging:
        staged = Path(staging) / DIRECTORY
        with zipfile.ZipFile(io.BytesIO(payload)) as package:
            package.extractall(staged)
        (staged / _MARKER).write_text(ARCHIVE_SHA256, encoding="utf-8")
        try:
            staged.rename(custom_nodes / DIRECTORY)
        except OSError:
            # Another setup may have published the same verified pack while we unpacked it.
            if not pack_installed(custom_nodes):
                raise
