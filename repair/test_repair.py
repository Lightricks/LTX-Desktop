"""Regression tests for the Windows repair ZIP and the isolated diagnostic runner."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from build import build
import repair_existing as repair

ROOT = Path(__file__).resolve().parent.parent


class RepairTests(unittest.TestCase):
    def test_archive_is_self_contained_and_reproducible(self):
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            archive = build(path)
            first = hashlib.sha256(archive.read_bytes()).hexdigest()
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()), {"Repair-LTX-Desktop-AMD.cmd", "Read-Me.txt",
                                                     "repair_existing.py", "repair_single_gpu.py", "diagnose.py"})
                cmd = z.read("Repair-LTX-Desktop-AMD.cmd")
                self.assertIn(b"\r\n", cmd)
                self.assertEqual(z.read("repair_single_gpu.py"), (ROOT / "runtimework/repair_single_gpu.py").read_bytes())
            self.assertEqual(first, hashlib.sha256(build(path).read_bytes()).hexdigest())

    def test_process_check_excludes_current_process_and_other_python_installations(self):
        exe = Path(sys.executable)
        def process(pid, name, path):
            return SimpleNamespace(info={"pid": pid, "name": name, "exe": str(path)})
        procs = [process(os.getpid(), "python.exe", exe), process(12, "python.exe", "/other/python.exe")]
        psutil = SimpleNamespace(process_iter=lambda _: procs)
        with mock.patch.dict(sys.modules, {"psutil": psutil}):
            repair.require_app_closed(exe)
            for busy in (process(13, "python.exe", exe), process(14, "LTX Desktop AMD.exe", "/some/path"),
                         process(15, "LTX Desktop - AMD 395 MAX.exe", "/some/path")):
                procs.append(busy)
                with self.assertRaisesRegex(RuntimeError, "Close LTX Desktop - AMD 395 MAX"):
                    repair.require_app_closed(exe)
                procs.pop()

    def test_supported_installs_are_recognized_under_both_executable_names(self):
        with tempfile.TemporaryDirectory() as base:
            application = Path(base)
            py = application / "resources/python/python.exe"
            for rel in ("resources/python/python.exe", "resources/backend/ltx2_server.py", "resources/python/ltx-amd-runtime.json"):
                entry = application / rel
                entry.parent.mkdir(parents=True, exist_ok=True)
                entry.touch()
            (application / repair.MARKER).write_text(repair.PRIOR_RELEASE, encoding="utf-8")
            with mock.patch.object(sys, "executable", str(py)):
                with self.assertRaisesRegex(RuntimeError, "Could not recognize"):
                    repair.check_install(application)
                for name in ("LTX Desktop AMD.exe", "LTX Desktop - AMD 395 MAX.exe"):
                    with self.subTest(name=name):
                        exe = application / name
                        exe.touch()
                        self.assertEqual(repair.check_install(application), py)
                        exe.unlink()

    def test_version_is_updated_only_for_an_owned_registry_record(self):
        with tempfile.TemporaryDirectory() as base:
            application = Path(base) / "application"
            application.mkdir()
            marker = application / repair.MARKER
            for version, uninstaller, expected_write in (
                (repair.PRIOR_RELEASE, application.parent / "Uninstall.exe", True),
                (repair.PRIOR_RELEASE, application.parent / "some-other-app.exe", False),
                ("future", application.parent / "Uninstall.exe", False),
            ):
                with self.subTest(version=version, uninstaller=uninstaller):
                    marker.write_text(repair.PRIOR_RELEASE, encoding="utf-8")
                    api = mock.MagicMock()
                    api.QueryValueEx.side_effect = [(version, 1), (f'"{uninstaller}"', 1)]
                    with mock.patch.dict(sys.modules, {"winreg": api}):
                        repair.record_release(application)
                    self.assertEqual(marker.read_text(), repair.RELEASE)
                    self.assertEqual(api.SetValueEx.called, expected_write)

    def test_isolated_diagnostic_reports_startup_and_failure(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            application = base / "Programs/LTXDesktopAMD/application"
            backend = application / "resources/backend"
            backend.mkdir(parents=True)
            server = backend / "ltx2_server.py"
            server.write_text("import os\nfrom pathlib import Path\n"
                              "assert os.environ['LTX_REQUIRE_ROCM'] == '1'\n"
                              "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
                              "assets = Path(os.environ['LTX_APP_DATA_DIR'])\n"
                              "assert assets != Path(os.environ['LOCALAPPDATA'])\n"
                              "print('TEMP-DATA', assets)\napp = lambda: None\n", encoding="utf-8")
            modules = {"transformers.py": "PreTrainedModel = ProcessorMixin = object",
                       "ltx_core/text_encoders/gemma.py": "LTXGemmaTextEncoder = object",
                       "ltx_pipelines/ic_lora.py": "", "av.py": "", "torchaudio.py": "", "torchvision.py": "",
                       "diffusers/hooks/context_parallel.py": "", "diffusers/models/_modeling_parallel.py": ""}
            for module, source in modules.items():
                directory = backend
                for part in Path(module).parts[:-1]:
                    directory /= part
                    directory.mkdir(exist_ok=True)
                    (directory / "__init__.py").touch()
                (backend / module).write_text(source, encoding="utf-8")
            def run():
                return subprocess.run([sys.executable, str(ROOT / "repair/diagnose.py"),
                                       "--application", str(application)], env=dict(os.environ, LOCALAPPDATA=str(base)),
                                      capture_output=True, text=True)
            ok = run()
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertIn("PASS:", ok.stdout)
            working = next(line.removeprefix("TEMP-DATA ") for line in ok.stdout.splitlines() if line.startswith("TEMP-DATA "))
            self.assertFalse(Path(working).exists())
            server.write_text("raise ImportError('simulated missing model dependency')", encoding="utf-8")
            failed = run()
            self.assertNotEqual(failed.returncode, 0)
            log = (base / "LTXDesktopAMD/logs/radeon-diagnostic.txt").read_text(encoding="utf-8")
            self.assertIn("ImportError: simulated missing", log)
            self.assertIn("FAIL:", failed.stdout)
            self.assertNotIn("PASS:", log)


if __name__ == "__main__":
    unittest.main()
