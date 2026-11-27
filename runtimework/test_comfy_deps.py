"""Check the private Windows engine dependency lock.

python runtimework/test_comfy_deps.py
python runtimework/test_comfy_deps.py --wheelhouse build/.cache/comfy-wheels --base-wheels BASE_WHEELHOUSE --cuda-python CUDA_PYTHON_DIR

Pass the extracted upstream resources/python directory to check the same private
wheel payload against the actual NVIDIA distribution and Windows dependency markers.
"""
from __future__ import annotations

import argparse
import copy
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import prepare_comfy_deps as deps

LOCK = json.loads(deps.MANIFEST.read_bytes())
BASE = json.loads(deps.BASE_MANIFEST.read_bytes())
VENDOR = deps.HERE.parent / "vendor" / "ComfyUI"
WHEELHOUSE = BASE_WHEELS = CUDA_PYTHON = None


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

        def locked_versions(lock):
            return dict(parse_wheel_filename(w["filename"])[:2] for w in lock["wheels"])

        overlay = locked_versions(LOCK)
        targets = {"rocm": locked_versions(BASE) | overlay}
        requirements = {"rocm": []}
        if CUDA_PYTHON is not None:
            site = CUDA_PYTHON / "Lib/site-packages"
            cuda = {canonicalize_name(d.metadata["Name"]): d
                    for d in importlib.metadata.distributions(path=[str(site)])}
            for name in ("torch", "torchvision", "torchaudio"):
                self.assertIn(name, cuda, f"Missing upstream NVIDIA {name} in {site}")
                self.assertIn("+cu", cuda[name].version, f"Expected NVIDIA build of {name}")
                self.assertNotIn(name, overlay, f"Private wheels override NVIDIA {name}")
            targets["cuda"] = {name: d.version for name, d in cuda.items()} | overlay
            requirements["cuda"] = [(name, requirement) for name in ("ltx-core", "ltx-pipelines")
                                    for requirement in cuda[name].requires or []]
        env = dict(default_environment(), python_version="3.13", python_full_version="3.13.12",
                   os_name="nt", sys_platform="win32", platform_machine="AMD64",
                   platform_system="Windows", implementation_version="3.13.12", extra="")

        def check(versions, source, requirement):
            r = Requirement(requirement)
            if r.marker and not r.marker.evaluate(env):
                return
            actual = versions.get(canonicalize_name(r.name))
            self.assertIsNotNone(actual, f"{source} dependency {r} missing")
            self.assertTrue(r.specifier.contains(actual, prereleases=True), f"{source}: {r}, locked {actual}")

        shared = []
        sources = [(WHEELHOUSE / w["filename"], w, shared) for w in LOCK["wheels"]]
        sources.extend((p, None, requirements["rocm"]) for p in (deps.HERE / "local-wheels").glob("ltx_*.whl"))
        for path, record, required in sources:
            with self.subTest(wheel=path.name):
                if record:
                    with path.open("rb") as src:
                        self.assertEqual(hashlib.file_digest(src, "sha256").hexdigest(), record["sha256"])
                with zipfile.ZipFile(path) as z:
                    m = next(i for i in z.namelist() if i.endswith(".dist-info/METADATA"))
                    for r in BytesParser().parsebytes(z.read(m)).get_all("Requires-Dist", []):
                        required.append((path.name, r))

        for filename in ("comfyui-v0.38.0.zip", "kandinsky6-1.0.2.zip"):
            with zipfile.ZipFile(VENDOR / filename) as z:
                reqs = min((n for n in z.namelist() if n.split("/")[-1] == "requirements.txt"), key=len)
                for line in z.read(reqs).decode().splitlines():
                    line = line.partition("#")[0].strip()
                    if line and not line.startswith(("comfyui-workflow-templates", "comfyui-embedded-docs")):
                        shared.append((filename, line))

        for target, versions in targets.items():
            for source, requirement in shared + requirements[target]:
                with self.subTest(target=target, source=source, requirement=requirement):
                    check(versions, source, requirement)

    def test_windows_gpu_and_media_files(self):
        if WHEELHOUSE is None:
            self.skipTest("pass --wheelhouse for the Windows GPU and video payload checks")
        paths = {deps.package_name(w["filename"]): WHEELHOUSE / w["filename"] for w in LOCK["wheels"]}
        with zipfile.ZipFile(paths["comfy-kitchen"]) as ck:
            self.assertIn(b"nvcuda.dll", ck.read("comfy_kitchen/backends/cuda/_C.abi3.pyd").lower())
            hip = "comfy_kitchen/backends/hip/"
            self.assertIn("gfx1151", json.loads(ck.read(hip + "architectures.json"))["wmma_gfx11"])
            binary = ck.read(hip + "_C.abi3.pyd")
            self.assertIn(b"hipv4-amdgcn-amd-amdhsa--gfx1151", binary)
            self.assertIn(b"amdhip64_7.dll", binary)
        with zipfile.ZipFile(paths["comfy-aimdo"]) as aimdo:
            self.assertIn(b"nvcuda.dll", aimdo.read("comfy_aimdo/aimdo.dll").lower())
            self.assertIn(b"amdhip64", aimdo.read("comfy_aimdo/aimdo_rocm.dll").lower())
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
    parser.add_argument("--cuda-python", type=Path, help="extracted official Windows CUDA interpreter directory")
    args = parser.parse_args()
    if (args.cuda_python or args.base_wheels) and not args.wheelhouse:
        parser.error("--base-wheels and --cuda-python require --wheelhouse")
    WHEELHOUSE, BASE_WHEELS, CUDA_PYTHON = args.wheelhouse, args.base_wheels, args.cuda_python
    unittest.main(argv=[__file__], verbosity=2)
