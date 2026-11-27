"""Negative checks for the release verifier. Run with python -m unittest discover -s checks."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from verify_install import ROOT, check_installed_runtime, check_manifest, inspect_shell


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
    def test_strix_halo_receipts_use_verified_gfx1151_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            resources = Path(d)
            (resources / "python").mkdir()
            canonical = (ROOT / "runtimework/runtime-manifest.json").read_bytes()
            (resources / "runtime-manifest.json").write_bytes(canonical)
            marker = {"target": "gfx1151", "manifest_sha256": hashlib.sha256(canonical).hexdigest()}
            (resources / "python/ltx-amd-runtime.json").write_text(json.dumps(marker))
            for model in ("8040S", "8050S", "8060S"):
                with self.subTest(model=model):
                    selection = {"version": 1, "backend": "rocm", "deviceName": f"AMD Radeon {model}", "hipOrdinal": 2}
                    (resources / "gpu-runtime.json").write_text(json.dumps(selection))
                    self.assertEqual(check_installed_runtime(resources, ROOT), selection)
            marker["target"] = "gfx1100"
            (resources / "python/ltx-amd-runtime.json").write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, "runtime target"):
                check_installed_runtime(resources, ROOT)

    def test_marker_and_installed_manifest_cannot_both_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected, installed, marker = (root / name for name in ("expected", "installed", "marker"))
            expected.write_bytes(b'{"target":"gfx1151"}')
            installed.write_bytes(b'{"target":"gfx1151", "modified":true}')
            marker.write_text(json.dumps({"target": "gfx1151", "manifest_sha256": hashlib.sha256(installed.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError, "differs from repository"):
                check_manifest(installed, marker, expected)

    def test_cuda_verifier_checks_upstream_pin_and_rejects_a_stale_amd_interpreter(self):
        with tempfile.TemporaryDirectory() as d:
            resources = Path(d)
            (resources / "python").mkdir()
            selection = {"version": 1, "backend": "cuda", "deviceName": "NVIDIA",
                         "cudaUuid": "GPU-11111111-3333-5555-9999-aabbccddeeff"}
            (resources / "gpu-runtime.json").write_text(json.dumps(selection))
            manifest = resources / "cuda-runtime.json"
            manifest.write_bytes((ROOT / "setupwork/src/cuda-runtime.json").read_bytes())
            self.assertEqual(check_installed_runtime(resources, ROOT)["backend"], "cuda")
            marker = resources / "python/ltx-amd-runtime.json"
            marker.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Radeon installation marker"):
                check_installed_runtime(resources, ROOT)
            marker.unlink()
            manifest.write_text('{"release":"unexpected"}')
            with self.assertRaisesRegex(ValueError, "differs from the pinned release"):
                check_installed_runtime(resources, ROOT)

    def test_api_cpu_verifier_requires_the_pinned_manifest_and_matching_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            resources = Path(temp)
            (resources / "python").mkdir()
            selection = {"version": 1, "backend": "api", "deviceName": "No supported local GPU"}
            (resources / "gpu-runtime.json").write_text(json.dumps(selection))
            canonical = (ROOT / "setupwork/src/api-runtime.json").read_bytes()
            manifest = resources / "api-runtime.json"
            manifest.write_bytes(canonical)
            receipt = {"target": "cpu-api", "manifest_sha256": hashlib.sha256(canonical).hexdigest()}
            marker = resources / "python/ltx-api-runtime.json"
            marker.write_text(json.dumps(receipt))
            self.assertEqual(check_installed_runtime(resources, ROOT), selection)
            (resources / "cuda-runtime.json").touch()
            with self.assertRaisesRegex(ValueError, "conflicting GPU"):
                check_installed_runtime(resources, ROOT)
            (resources / "cuda-runtime.json").unlink()
            receipt["target"] = "gfx1151"
            marker.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "runtime target"):
                check_installed_runtime(resources, ROOT)

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
