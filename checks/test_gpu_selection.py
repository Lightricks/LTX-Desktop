"""Windows installer device selection. Exercised on Linux and Windows without drivers or weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setupwork/src"))
import gpu_selection as gpu

UUID = "GPU-11111111-3333-5555-9999-aabbccddeeff"
GIB = 1024**3
RADEON = {"Name": "AMD Radeon(TM) Graphics", "PNPDeviceID": r"PCI\VEN_1002&DEV_1586&REV_C1"}


def choose(nvidia=(), windows=(), **kwargs):
    with patch.object(gpu, "_cuda_adapters", return_value=nvidia), patch.object(
        gpu, "_windows_display_adapters", return_value=windows
    ):
        return gpu.select_gpu(**kwargs)


class WindowsGpuSelection(unittest.TestCase):
    def test_cuda_wins_when_both_supported_and_pins_actual_adapter(self):
        selected = choose([(8*GIB, (8, 9), "display", "other-uuid"), (24*GIB, (8, 6), "RTX 3090", UUID)], [RADEON])
        self.assertEqual(selected, {"version": 1, "backend": "cuda", "deviceName": "RTX 3090", "cudaUuid": UUID})
        env = gpu.child_environment(selected, {"HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": UUID, "LTX_GPU_RUNTIME": "cuda", "LTX_REQUIRE_ROCM": "0"})

    def test_upstream_windows_capacity_threshold_and_amd_fallback(self):
        for capacity, sm, expected in ((9*GIB, (8, 9), "cuda"), (9*GIB-1, (8, 9), "rocm"),
                                       (15*GIB, (8, 6), "cuda"), (15*GIB-1, (8, 6), "rocm")):
            with self.subTest(capacity=capacity, sm=sm):
                self.assertEqual(choose([(capacity, sm, "NVIDIA", UUID)], [RADEON])["backend"], expected)
        self.assertEqual(choose([(24*GIB, (8, 6), "NVIDIA", UUID)], [RADEON], exclude_cuda=True)["backend"], "rocm")

    def test_unknown_radeon_or_intel_uses_api_only(self):
        selected = choose(windows=[{"Name": "AMD Radeon RX 7900", "PNPDeviceID": r"PCI\VEN_1002&DEV_744C"}])
        self.assertEqual(selected["backend"], "api")
        self.assertEqual(gpu.child_environment(selected, {})["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(choose()["backend"], "api")

    def test_rocm_binds_kernel_architecture_and_both_visibility_aliases(self):
        selected = choose(windows=[RADEON])
        completed = subprocess.CompletedProcess([], 0, json.dumps([[0, "gfx1100", "Radeon RX"],
                                                                    [1, "gfx1151", "AMD Radeon 8060S"]]))
        with patch.object(subprocess, "run", return_value=completed):
            gpu.bind_rocm_device(Path("python.exe"), selected)
        self.assertEqual(selected["hipOrdinal"], 1)
        self.assertEqual(gpu.child_environment(selected, {"rocr_visible_devices": "0"}),
                         {"LTX_GPU_RUNTIME": "rocm", "LTX_REQUIRE_ROCM": "1",
                          "CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"})

    def test_probe_does_not_activate_unusable_radeon(self):
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "[]")):
            with self.assertRaisesRegex(RuntimeError, "ROCm could not use it"):
                gpu.bind_rocm_device(Path("python.exe"), choose(windows=[RADEON]))

    def test_receipt_requires_valid_identity_even_if_legacy_marker_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "python").mkdir()
            (root / "python/ltx-amd-runtime.json").write_text("{}")
            self.assertEqual(gpu.read_selection(root)["backend"], "rocm")
            bad = {"version": 1, "backend": "cuda", "cudaUuid": "", "deviceName": "GeForce"}
            (root / "gpu-runtime.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                gpu.read_selection(root)
            bad["cudaUuid"] = UUID
            (root / "gpu-runtime.json").write_text(json.dumps(bad))
            self.assertEqual(gpu.read_selection(root), bad)


if __name__ == "__main__":
    unittest.main()
