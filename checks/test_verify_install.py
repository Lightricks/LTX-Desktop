"""Negative checks for the release verifier. Run with python -m unittest discover -s checks."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from verify_install import check_manifest, inspect_shell


def build_fixture(path, files):
    header = {}
    data = bytearray()
    for name, content in files.items():
        parts = name.split("/")
        node = header
        for part in parts[:-1]:
            node = node.setdefault("files", {}).setdefault(part, {})
        node.setdefault("files", {})[parts[-1]] = {
            "offset": str(len(data)), "size": len(content),
            "integrity": {"algorithm": "SHA256", "hash": hashlib.sha256(content).hexdigest()},
        }
        data.extend(content)
    payload = json.dumps(header).encode()
    padding = b"\0" * (-len(payload) % 4)
    size = len(payload) + len(padding)
    path.write_bytes(struct.pack("<IIII", 4, size + 8, size + 4, len(payload)) + payload + padding + data)


class ReleaseVerifierTests(unittest.TestCase):
    def test_marker_and_installed_manifest_cannot_both_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected, installed, marker = (root / name for name in ("expected", "installed", "marker"))
            expected.write_bytes(b'{"target":"gfx1151"}')
            installed.write_bytes(b'{"target":"gfx1151", "modified":true}')
            marker.write_text(json.dumps({"target": "gfx1151", "manifest_sha256": hashlib.sha256(installed.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError, "differs from repository"):
                check_manifest(installed, marker, expected)

    def test_fresh_shell_passes_but_missing_or_damaged_renderer_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "app.asar"
            files = {
                "package.json": b'{"name":"ltx-desktop-amd","productName":"LTX Desktop - AMD 395 MAX","main":"dist-electron/main.js"}',
                "dist-electron/main.js": b"new main",
                "dist-electron/preload.js": b"new preload",
                "dist/index.html": b'<script type="module" src="./assets/new.js"></script>',
                "dist/assets/new.js": b"new renderer",
            }
            build_fixture(path, files)
            inspect_shell(path)
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + b"!")
            with self.assertRaisesRegex(ValueError, "Corrupt Electron"):
                inspect_shell(path)
            files.pop("dist/assets/new.js")
            build_fixture(path, files)
            with self.assertRaises(KeyError):
                inspect_shell(path)
