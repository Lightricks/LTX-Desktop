"""Check the private Windows engine dependency lock.

python runtimework/test_comfy_deps.py
python runtimework/test_comfy_deps.py --wheelhouse build/.cache/comfy-wheels --base-wheels BASE_WHEELHOUSE

The second command also checks each installed dependency requirement using the
packaging module on the release builder, including Windows-specific markers.
"""
from __future__ import annotations

import argparse
import copy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import prepare_comfy_deps as deps

LOCK = json.loads(deps.MANIFEST.read_bytes())
BASE = json.loads(deps.BASE_MANIFEST.read_bytes())
VENDOR = deps.HERE.parent / "vendor" / "ComfyUI"
WHEELHOUSE = BASE_WHEELS = None


class PrivateDependencyTests(unittest.TestCase):
    def test_protected_runtime(self):
        deps.validate_lock(LOCK, BASE)
        bad = copy.deepcopy(LOCK)
        torch = next(w for w in BASE["wheels"] if deps.package_name(w["filename"]) == "torch")
        bad["wheels"].append(dict(torch, url="https://files.pythonhosted.org/test.whl"))
        with self.assertRaisesRegex(ValueError, "protected distribution torch"):
            deps.validate_lock(bad, BASE)

    def test_win64_python_abi(self):
        def check(tags):
            return deps.compatible_tags(f"demo-1.0-{tags}.whl", "3.13.12")
        self.assertTrue(check("cp312-abi3-win_amd64"))
        self.assertTrue(check("cp313-cp313-win_amd64"))
        self.assertTrue(check("py2.py3-none-any"))
        self.assertFalse(check("cp314-abi3-win_amd64"))
        self.assertFalse(check("cp312-cp312-win_amd64"))
        self.assertFalse(check("cp313-cp313t-win_amd64"))
        self.assertFalse(check("py3-none-win_arm64"))

    def test_paths_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wheel = tmp / "test.whl"
            site = tmp / "output"
            site.mkdir()
            with zipfile.ZipFile(wheel, "w") as z:
                z.writestr("demo-1.0.data/purelib/demo/__init__.py", "value = 1\n")
            deps.unpack(wheel, site)
            self.assertTrue((site / "demo/__init__.py").is_file())
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                deps.unpack(wheel, site)
            with zipfile.ZipFile(wheel, "w") as z:
                z.writestr("../outside", "content")
            with self.assertRaisesRegex(ValueError, "Invalid wheel entry"):
                deps.unpack(wheel, site)
            self.assertFalse((tmp / "outside").exists())

    def test_real_dependency_closure(self):
        if WHEELHOUSE is None:
            self.skipTest("pass --wheelhouse for the complete release-wheel check")
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name, parse_wheel_filename

        # All three GPU packages have AMD local-version metadata which the
        # public PyPI index cannot resolve or accurately substitute.
        versions = {}
        for lock in (BASE, LOCK):
            for record in lock["wheels"]:
                name, version, _, _ = parse_wheel_filename(record["filename"])
                versions[name] = version
        env = dict(default_environment(), python_version="3.13", python_full_version="3.13.12",
                   os_name="nt", sys_platform="win32", platform_machine="AMD64",
                   platform_system="Windows", implementation_version="3.13.12", extra="")

        def check(source, requirement):
            r = Requirement(requirement)
            if r.marker and not r.marker.evaluate(env):
                return
            actual = versions.get(canonicalize_name(r.name))
            self.assertIsNotNone(actual, f"{source} dependency {r} missing")
            self.assertTrue(r.specifier.contains(actual, prereleases=True), f"{source}: {r}, locked {actual}")

        sources = [(WHEELHOUSE / w["filename"], w) for w in LOCK["wheels"]]
        sources.extend((p, None) for p in (deps.HERE / "local-wheels").glob("ltx_*.whl"))
        for path, record in sources:
            with self.subTest(wheel=path.name):
                if record:
                    with path.open("rb") as src:
                        self.assertEqual(hashlib.file_digest(src, "sha256").hexdigest(), record["sha256"])
                with zipfile.ZipFile(path) as z:
                    m = next(i for i in z.namelist() if i.endswith(".dist-info/METADATA"))
                    for r in BytesParser().parsebytes(z.read(m)).get_all("Requires-Dist", []):
                        check(path.name, r)

        for filename in ("comfyui-v0.38.0.zip", "kandinsky6-1.0.2.zip"):
            with zipfile.ZipFile(VENDOR / filename) as z:
                reqs = min((n for n in z.namelist() if n.split("/")[-1] == "requirements.txt"), key=len)
                for line in z.read(reqs).decode().splitlines():
                    line = line.partition("#")[0].strip()
                    if line and not line.startswith(("comfyui-workflow-templates", "comfyui-embedded-docs")):
                        check(filename, line)

    def test_windows_gpu_and_media_files(self):
        if WHEELHOUSE is None:
            self.skipTest("pass --wheelhouse for the Windows GPU and video payload checks")
        paths = {deps.package_name(w["filename"]): WHEELHOUSE / w["filename"] for w in LOCK["wheels"]}
        with zipfile.ZipFile(paths["comfy-kitchen"]) as ck:
            hip = "comfy_kitchen/backends/hip/"
            self.assertIn("gfx1151", json.loads(ck.read(hip + "architectures.json"))["wmma_gfx11"])
            binary = ck.read(hip + "_C.abi3.pyd")
            self.assertIn(b"hipv4-amdgcn-amd-amdhsa--gfx1151", binary)
            self.assertIn(b"amdhip64_7.dll", binary)
        with zipfile.ZipFile(paths["av"]) as av:
            names = {p.casefold() for p in av.namelist() if p.casefold().endswith(".dll")}
            for prefix in ("avcodec-", "avformat-", "swresample-", "libx264-"):
                self.assertTrue(any(p.startswith("av.libs/" + prefix) for p in names), prefix)
            if BASE_WHEELS is not None:
                base_av = next(w for w in BASE["wheels"] if deps.package_name(w["filename"]) == "av")
                with zipfile.ZipFile(BASE_WHEELS / base_av["filename"]) as native:
                    common = names & {x.casefold() for x in native.namelist()}
                    for name in common:
                        original = next(x for x in native.namelist() if x.casefold() == name)
                        private = next(x for x in av.namelist() if x.casefold() == name)
                        self.assertEqual(native.read(original), av.read(private), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--base-wheels", type=Path)
    args = parser.parse_args()
    WHEELHOUSE, BASE_WHEELS = args.wheelhouse, args.base_wheels
    unittest.main(argv=[__file__], verbosity=2)
