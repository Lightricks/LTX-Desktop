"""Check a staged or installed LTX Desktop - AMD 395 MAX build against this source tree.

Run with Python 3.11+. On the Windows target, --gpu additionally runs the
installed runtime's device checks. On Radeon it also checks the patched LTX operators.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = Path(__file__).with_name("release-shell.json")
sys.path.insert(0, str(ROOT / "setupwork/src"))
from gpu_selection import child_environment, read_selection


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_bytes(path):
    # Git for Windows may check out the same source with CRLF endings.
    return path.read_bytes().replace(b"\r\n", b"\n")


def check_sources(source, destination, patterns):
    if not source.is_dir():
        raise ValueError(f"Missing source directory: {source}")
    total = 0
    for pattern in patterns:
        for path in source.rglob(pattern):
            relative = path.relative_to(source)
            if any(part in {"tests", "__pycache__", ".venv"} for part in relative.parts):
                continue
            installed = destination / relative
            if not installed.is_file() or text_bytes(path) != text_bytes(installed):
                raise ValueError(f"Source differs from installed file: {installed}")
            total += 1
    if not total:
        raise ValueError(f"Empty source check: {source}")
    return total


def inspect_shell(path, original_release=False):
    """Read the Chromium pickle header and the actual Electron entry point bytes."""
    with path.open("rb") as stream:
        prefix = stream.read(16)
        if len(prefix) != 16:
            raise ValueError("Truncated Electron archive header")
        word, header_size, inner_size, json_size = struct.unpack("<IIII", prefix)
        if word != 4 or inner_size + 4 != header_size or not 0 < json_size <= header_size - 8 <= 32 * 1024**2:
            raise ValueError("Invalid Electron archive header")
        tree = json.loads(stream.read(json_size))
        data_start = 8 + header_size
        archive_size = path.stat().st_size

        def read_entry(name):
            node = tree
            for part in name.split("/"):
                node = node["files"][part]
            offset = data_start + int(node["offset"])
            size = int(node["size"])
            if offset < data_start or size <= 0 or size > 128 * 1024**2 or offset + size > archive_size:
                raise ValueError(f"Invalid span for Electron archive entry: {name}")
            stream.seek(offset)
            content = stream.read(size)
            integrity = node.get("integrity")
            if integrity and integrity["algorithm"] == "SHA256":
                if hashlib.sha256(content).hexdigest() != integrity["hash"]:
                    raise ValueError(f"Corrupt Electron archive entry: {name}")
            return content

        package = json.loads(read_entry("package.json"))
        if package.get("name") != "ltx-desktop-amd":
            raise ValueError("The archive does not identify the AMD desktop application")
        if not original_release and package.get("productName") != "LTX Desktop - AMD 395 MAX":
            raise ValueError("Incorrect Windows desktop product name in the application archive")
        read_entry(package["main"].removeprefix("./"))
        read_entry("dist-electron/preload.js")
        html = read_entry("dist/index.html").decode("utf-8")
        modules = re.findall(r'<script\b[^>]*?\bsrc=["\'](\./assets/[^"\']+\.js)["\']', html)
        if not modules:
            raise ValueError("Missing compiled renderer script in dist/index.html")
        for module in modules:
            read_entry("dist/" + module.removeprefix("./"))


def check_manifest(runtime_manifest, marker_path, expected_manifest):
    marker = json.loads(marker_path.read_bytes())
    if marker.get("target") != "gfx1151" or sha256(runtime_manifest) != marker.get("manifest_sha256"):
        raise ValueError("AMD runtime target or manifest hash does not match")
    payload = runtime_manifest.read_bytes()
    if payload != expected_manifest.read_bytes():
        raise ValueError("Installed runtime manifest differs from repository/runtimework/runtime-manifest.json")
    if json.loads(payload).get("target") != marker["target"]:
        raise ValueError("Target mismatch between runtime manifest and runtime marker")


def check_installed_runtime(resources, source):
    selection = read_selection(resources)
    if selection["backend"] == "rocm":
        check_manifest(resources / "runtime-manifest.json", resources / "python/ltx-amd-runtime.json",
                       source / "runtimework/runtime-manifest.json")
    else:
        manifest = resources / "cuda-runtime.json"
        expected = source / "setupwork/src/cuda-runtime.json"
        # JSON equality tolerates a Git-for-Windows checkout with CRLF line endings.
        if json.loads(manifest.read_bytes()) != json.loads(expected.read_bytes()):
            raise ValueError("NVIDIA runtime manifest differs from the pinned release")
        if (resources / "python/ltx-amd-runtime.json").exists():
            raise ValueError("The NVIDIA runtime contains a Radeon installation marker")
    return selection


def verify(application, root, run_gpu, original_release):
    resources = application / "resources"
    embedded = resources / "python"
    executable = embedded / "python.exe"
    launcher = application / ("LTX Desktop AMD.exe" if original_release else "LTX Desktop - AMD 395 MAX.exe")
    for path in (launcher, executable):
        with path.open("rb") as stream:
            if stream.read(2) != b"MZ":
                raise ValueError(f"Invalid Windows executable: {path}")
    selection = check_installed_runtime(resources, root)
    backend = selection["backend"]
    shell = resources / "app.asar"
    inspect_shell(shell, original_release)
    if not original_release:
        engine = resources / "comfy-engine"
        for archive in ("comfyui-v0.38.0.zip", "kandinsky6-1.0.2.zip"):
            if sha256(engine / archive) != sha256(root / "vendor/ComfyUI" / archive):
                raise ValueError(f"Bundled renderer archive differs from the source distribution: {archive}")
        receipt = json.loads((engine / "site-packages/ltx-comfy-deps.json").read_bytes())
        if receipt["manifest_sha256"] != sha256(root / "runtimework/comfy-windows-wheels.json"):
            raise ValueError("Private renderer dependencies differ from the Windows release lock")
    if original_release:
        receipt = json.loads(RECEIPT.read_bytes())
        if sha256(shell) != receipt["app_asar_sha256"]:
            raise ValueError("The installed shell differs from the originally distributed shell")
        for rel, expected in receipt["desktop_sources"].items():
            if hashlib.sha256(text_bytes(root / rel)).hexdigest() != expected:
                raise ValueError(f"AMD desktop source differs from original release: {rel}")
        print("PASS: original compiled shell and AMD desktop source hashes.", flush=True)

    count = check_sources(root / "app/backend", resources / "backend", ("*.py", "*.sql", "*.json", "pyproject.toml"))
    count += check_sources(root / "app/shared", resources / "shared", ("*.json",))
    packages = root / "vendor/LTX-2/packages"
    site = embedded / "Lib/site-packages"
    for package, module in (("ltx-core", "ltx_core"), ("ltx-pipelines", "ltx_pipelines")):
        if backend == "rocm":
            count += check_sources(packages / package / "src" / module, site / module, ("*.py",))
        elif not (site / module / "__init__.py").is_file():
            raise ValueError(f"Missing inference package in upstream Python runtime: {module}")
    diagnostics = ["AMD-Diagnostics.cmd", "diagnose.py"]
    if not original_release:
        diagnostics.append("gpu_selection.py")
    for name in diagnostics:
        if text_bytes(root / "setupwork/src" / name) != text_bytes(application / name):
            raise ValueError(f"Diagnostic differs from source: {name}")
    print(f"PASS: {backend} runtime manifest, compiled Electron and renderer entry points, diagnostics, "
          f"and {count} backend, model and shared source files.", flush=True)

    if run_gpu:
        if os.name != "nt":
            raise ValueError("--gpu requires the installed Windows runtime")
        env = child_environment(selection)
        subprocess.run([str(executable), "-u", "-X", "utf8", str(application / "diagnose.py")], env=env, check=True)
        if backend == "rocm":
            subprocess.run([str(executable), "-u", "-X", "utf8", str(root / "vendor/LTX-2/radeon/check_ops.py"),
                            "--device", "cuda"], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, help="Install directory or its application subdirectory")
    parser.add_argument("--source", type=Path, default=ROOT, help="Complete release source checkout")
    parser.add_argument("--gpu", action="store_true", help="Run installed GPU diagnostics (API-only installs verify startup)")
    parser.add_argument("--original-release", action="store_true", help="Also require the original distributed desktop shell and AMD source hashes")
    args = parser.parse_args()
    base = args.install
    if base is None:
        if not os.environ.get("LOCALAPPDATA"):
            parser.error("Supply --install when running outside Windows.")
        base = Path(os.environ["LOCALAPPDATA"]) / "Programs/LTXDesktopAMD"
    base = base.resolve()
    application = base / "application" if (base / "application").is_dir() else base
    try:
        verify(application, args.source.resolve(), args.gpu, args.original_release)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        sys.exit(f"FAIL: {exc}")


if __name__ == "__main__":
    main()
