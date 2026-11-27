"""Regression tests for provisioning the correct Windows runtime on each hardware profile."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import cuda_runtime, gpu_selection, setup_engine

CUDA = {"version": 1, "backend": "cuda", "deviceName": "NVIDIA GPU", "cudaUuid": "GPU-test"}
AMD = {"version": 1, "backend": "rocm", "deviceName": "AMD Radeon 8060S"}
API = {"version": 1, "backend": "api", "deviceName": "No supported local GPU"}


def stage_release(root, members):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, data in members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    blob = gzip.compress(raw.getvalue())
    parts = []
    cache = root / "cache" / "cuda-v1.3.3"
    cache.mkdir(parents=True)
    for i, data in enumerate((blob[:17], blob[17:])):
        name = f"runtime.part-{i}"
        (cache / name).write_bytes(data)
        parts.append({"filename": name, "url": "https://example.invalid/" + name,
                      "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"archiveRoot": "python-embed", "release": "v1.3.3", "parts": parts,
                "unpackedBytes": sum(map(len, members.values()))}
    (root / "cuda-runtime.json").write_text(json.dumps(manifest), encoding="utf-8")


class WindowsGpuInstallerTests(unittest.TestCase):
    def test_cuda_and_api_use_exact_split_runtime_with_no_rocm_repair(self):
        for selected in (CUDA, API):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                resources = root / "resources"
                resources.mkdir()
                stage_release(root, {"python-embed/python.exe": b"official",
                                     "python-embed/Lib/site-packages/torch/version.py": b"CUDA"})
                with patch.object(cuda_runtime, "check_python") as probe, patch.object(setup_engine, "install_rocm") as rocm:
                    downloads = setup_engine.install_runtime_for_hardware(root, resources, root, root / "cache", selected)
                rocm.assert_not_called()
                python_dir, selected_backend, env = probe.call_args.args
                self.assertEqual(python_dir, resources / "python")
                self.assertEqual(selected_backend["backend"], selected["backend"])
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], selected.get("cudaUuid", "-1"))
                self.assertEqual(len(downloads), 2)
                self.assertEqual((python_dir / "python.exe").read_bytes(), b"official")
                self.assertEqual((resources / "cuda-runtime.json").read_bytes(), (root / "cuda-runtime.json").read_bytes())
                self.assertEqual(json.loads((resources / "gpu-runtime.json").read_text()), selected)
                self.assertFalse((resources / "runtime-manifest.json").exists())

    def test_rocm_does_not_fetch_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            resources = Path(temp)
            with patch.object(cuda_runtime, "install") as cuda, patch.object(setup_engine, "install_rocm", return_value=[]) as rocm, \
                    patch.object(gpu_selection, "bind_rocm_device") as bind:
                setup_engine.install_runtime_for_hardware(resources, resources, resources, resources, dict(AMD))
            cuda.assert_not_called()
            rocm.assert_called_once()
            bind.assert_called_once()
            self.assertEqual(json.loads((resources / "gpu-runtime.json").read_bytes())["backend"], "rocm")

    def test_failed_cuda_probe_uses_verified_radeon_alternative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = root / "resources"
            resources.mkdir()
            stage_release(root, {"python-embed/python.exe": b"cuda-only"})
            with patch.object(cuda_runtime, "check_python", side_effect=RuntimeError("CUDA driver failed")), \
                    patch.object(gpu_selection, "select_gpu", return_value=dict(AMD)) as select, \
                    patch.object(setup_engine, "install_rocm", return_value=[]) as rocm, \
                    patch.object(gpu_selection, "bind_rocm_device"), self.assertLogs(level="WARNING"):
                setup_engine.install_runtime_for_hardware(root, resources, root, root / "cache", dict(CUDA))
            rocm.assert_called_once()
            select.assert_called_once_with(exclude_cuda=True)
            self.assertFalse((resources / "python/python.exe").exists(), "CUDA staging must be cleared before ROCm")
            self.assertFalse((resources / "cuda-runtime.json").exists())
            self.assertEqual(json.loads((resources / "gpu-runtime.json").read_bytes())["backend"], "rocm")

    def test_failed_cuda_on_nvidia_only_reports_driver_and_does_not_select_paid_api(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = root / "resources"
            resources.mkdir()
            stage_release(root, {"python-embed/python.exe": b"official"})
            with patch.object(cuda_runtime, "check_python", side_effect=RuntimeError("driver unavailable")), \
                    patch.object(gpu_selection, "select_gpu", return_value=dict(API)):
                with self.assertRaisesRegex(RuntimeError, "Update the NVIDIA driver"):
                    setup_engine.install_runtime_for_hardware(root, resources, root, root / "cache", dict(CUDA))
            self.assertFalse((resources / "gpu-runtime.json").exists())

    def test_rejects_archive_escape_and_truncated_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage_release(root, {"python-embed/../outside": b"invalid"})
            paths = sorted((root / "cache/cuda-v1.3.3").iterdir())
            with self.assertRaisesRegex(ValueError, "Unexpected path"):
                cuda_runtime.extract(paths, root / "python", "python-embed")
            self.assertFalse((root / "outside").exists())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage_release(root, {"python-embed/python.exe": b"ok"})
            paths = sorted((root / "cache/cuda-v1.3.3").iterdir())
            paths[-1].write_bytes(paths[-1].read_bytes()[:-2])
            with self.assertRaises(EOFError):
                cuda_runtime.extract(paths, root / "python", "python-embed")

    def test_failed_runtime_probe_leaves_existing_install_active(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = root / "application"
            current.mkdir()
            (current / setup_engine.MARKER).write_text("older-release", encoding="utf-8")
            (current / "old-file").write_bytes(b"preserve")
            compact = Mock()
            compact.required_bytes.return_value = 1000
            with patch.object(gpu_selection, "select_gpu", return_value=dict(CUDA)), \
                    patch.object(setup_engine, "install_runtime_for_hardware", side_effect=RuntimeError("GPU unavailable")):
                with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                    setup_engine.deploy(root, root, root, root, compact)
            self.assertEqual((current / "old-file").read_bytes(), b"preserve")
            self.assertFalse((root / "installing").exists())


if __name__ == "__main__":
    unittest.main()
