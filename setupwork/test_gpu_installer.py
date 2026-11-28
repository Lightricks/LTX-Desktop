"""Regression tests for provisioning the correct Windows runtime on each hardware profile."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile
import shutil
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
    def test_cuda_uses_official_split_runtime_and_app_local_crt(self):
        selected = CUDA
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = root / "resources"
            resources.mkdir()
            stage_release(root, {"python-embed/python.exe": b"official",
                                 "python-embed/Lib/site-packages/torch/version.py": b"CUDA"})
            (root / "vcredist").mkdir()
            (root / "vcredist" / "msvcp140.dll").write_bytes(b"CRT")
            with patch.object(cuda_runtime, "check_python") as probe, patch.object(setup_engine, "install_rocm") as rocm:
                downloads = setup_engine.install_runtime_for_hardware(root, resources, root, root / "cache", selected)
            rocm.assert_not_called()
            python_dir, selected_backend, env = probe.call_args.args
            self.assertEqual(python_dir, resources / "python")
            self.assertEqual(selected_backend["backend"], selected["backend"])
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], selected["cudaUuid"])
            self.assertEqual(len(downloads), 2)
            self.assertEqual((python_dir / "python.exe").read_bytes(), b"official")
            self.assertEqual((python_dir / "msvcp140.dll").read_bytes(), b"CRT")
            self.assertEqual((resources / "cuda-runtime.json").read_bytes(), (root / "cuda-runtime.json").read_bytes())
            self.assertEqual(json.loads((resources / "gpu-runtime.json").read_text()), selected)
            self.assertFalse((resources / "runtime-manifest.json").exists())

    def test_api_stages_cpu_wheels_and_validates_server_before_writing_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = root / "resources"
            resources.mkdir()
            runtime = root / "python"
            runtime.mkdir()
            (runtime / "python.exe").write_bytes(b"windows fixture")
            crt = root / "vcredist"
            crt.mkdir()
            (crt / "msvcp140.dll").write_bytes(b"CRT")
            repo = Path(__file__).resolve().parents[1]
            shutil.copy(repo / "runtimework/install_runtime.py", root)
            packages = root / "wheels"
            packages.mkdir()
            wheel = packages / "test_package-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("test_package/example.py", "value = 1")
            manifest = {"python": "3.13.12", "target": "cpu-api", "wheels": [{
                "filename": wheel.name, "sha256": setup_engine.digest(wheel),
                "size": wheel.stat().st_size, "url": None}]}
            (root / "api-runtime.json").write_text(json.dumps(manifest))
            def verify(stage, env):
                self.assertEqual(env["LTX_GPU_RUNTIME"], "api")
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
                self.assertFalse((stage / "gpu-runtime.json").exists())
                self.assertFalse((stage / "cuda-runtime.json").exists())
                self.assertFalse((stage / "python/ltx-amd-runtime.json").exists())
                self.assertFalse((stage / "python/Lib/site-packages/sitecustomize.py").exists())
                self.assertEqual((stage / "python/msvcp140.dll").read_bytes(), b"CRT")
                self.assertEqual(json.loads((stage / "python/ltx-api-runtime.json").read_bytes())["target"], "cpu-api")
                self.assertTrue((stage / "python/Lib/site-packages/test_package/example.py").is_file())
            with patch.object(setup_engine, "check_api_runtime", side_effect=verify) as check, patch.object(cuda_runtime, "install") as cuda:
                setup_engine.install_runtime_for_hardware(root, resources, packages, root / "cache", dict(API))
            check.assert_called_once()
            cuda.assert_not_called()
            self.assertEqual(json.loads((resources / "gpu-runtime.json").read_bytes()), API)
            self.assertEqual(json.loads((resources / "api-runtime.json").read_bytes()), manifest)

    def test_api_probe_failure_keeps_the_selection_unpublished(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(setup_engine, "install_locked_wheels", return_value=[]), \
                    patch.object(setup_engine, "check_api_runtime", side_effect=RuntimeError("CPU import failed")):
                with self.assertRaisesRegex(RuntimeError, "CPU import failed"):
                    setup_engine.install_runtime_for_hardware(root, root, root, root, dict(API))
            self.assertFalse((root / "gpu-runtime.json").exists())

    def test_api_lock_uses_common_app_wheels_and_only_cpu_torch(self):
        repo = Path(__file__).resolve().parents[1]
        def records(path):
            return {w["filename"].split("-")[0]: w for w in json.loads(path.read_bytes())["wheels"]}
        cpu = records(repo / "setupwork/src/api-runtime.json")
        rocm = records(repo / "runtimework/runtime-manifest.json")
        for name in ("rocm", "rocm_sdk_core", "rocm_sdk_libraries_gfx1151"):
            del rocm[name]
        for name in ("torch", "torchvision", "torchaudio"):
            del rocm[name]
            self.assertIn("+cpu-cp313-cp313-win_amd64.whl", cpu.pop(name)["filename"])
        self.assertEqual(cpu, rocm)

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

    def test_unusable_amd_candidate_does_not_write_a_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = dict(AMD, deviceName="AMD Radeon 8050S")
            with patch.object(setup_engine, "install_rocm", return_value=[]), \
                    patch.object(gpu_selection, "bind_rocm_device", side_effect=RuntimeError("GPU probe failed")):
                with self.assertRaisesRegex(RuntimeError, "GPU probe failed"):
                    setup_engine.install_runtime_for_hardware(root, root, root, root, candidate)
            self.assertFalse((root / "gpu-runtime.json").exists())

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
