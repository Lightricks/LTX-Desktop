#!/usr/bin/env python3
"""Build and verify the Windows x64 offline and online installers on Linux."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from nsis import select_nsis
from src.build_desktop import build as build_desktop
from verify import sha256, verify_installer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "setupwork/src"
FILENAME = {
    "offline": "LTX-Desktop-AMD-395-MAX-1.3.2-Windows-x64-Setup.exe",
    "web": "LTX-Desktop-AMD-395-MAX-Online-Setup.exe",
}


def require_files(folder, names):
    for name in names:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def compress_runtime(runtime, output, threads):
    """Keep memory use bounded when compressing the several-GiB Windows DLL tree."""
    if any(path.is_symlink() for path in runtime.rglob("*")):
        raise ValueError("The embedded Windows runtime must contain only regular files and directories")
    total = sum(path.stat().st_size for path in runtime.rglob("*") if path.is_file())
    tar_command = ["tar", "--sort=name", "--mtime=@0", "--numeric-owner", "--owner=0", "--group=0",
                   "--hard-dereference", "-C", str(runtime), "-cf", "-", "."]
    with subprocess.Popen(tar_command, stdout=subprocess.PIPE) as tar, output.open("wb") as dest:
        try:
            subprocess.run(["xz", "-1", f"-T{threads}", "--block-size=16MiB", "-c"],
                           stdin=tar.stdout, stdout=dest, check=True)
        finally:
            tar.stdout.close()
        if tar.wait():
            raise RuntimeError("tar failed archiving the Windows runtime")
    return total


def stage_stub_files(stage, runtime, mode, media, manifest, runtime_size):
    inputs = {"runtime-manifest.json": runtime / "runtime-manifest.json",
              "sitecustomize.py": runtime / "sitecustomize.py",
              "repair_single_gpu.py": runtime / "repair_single_gpu.py"}
    for dirname in ("python", "vcredist"):
        for path in (runtime / dirname).rglob("*"):
            if path.is_file():
                inputs[path.relative_to(runtime).as_posix()] = path
    if mode == "web":
        inputs["install_runtime.py"] = runtime / "install_runtime.py"
        for record in manifest["wheels"]:
            if record["url"]:
                continue
            wheel = runtime / "wheels" / record["filename"]
            if wheel.name != record["filename"] or wheel.stat().st_size != record["size"] or sha256(wheel) != record["sha256"]:
                raise ValueError(f"Missing or damaged private wheel: {wheel}")
            inputs["wheels/" + wheel.name] = wheel
    for name, path in inputs.items():
        dest = stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    index = {p.name: {"sha256": sha256(p), "size": p.stat().st_size} for p in media}
    index_name = "offline-index.json" if mode == "offline" else "online-index.json"
    write_json(stage / index_name, index)
    inputs[index_name] = stage / index_name
    if mode == "offline":
        size_path = stage / "runtime-size.json"
        write_json(size_path, {"unpacked_bytes": runtime_size})
        inputs[size_path.name] = size_path
    for name in ("setup_engine.py", "diagnose.py", "AMD-Diagnostics.cmd"):
        inputs[name] = SRC / name
    return inputs, index


def seal_setup(stub, output, media):
    with output.open("wb") as dest:
        with stub.open("rb") as program:
            shutil.copyfileobj(program, dest, 2**20)
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in media:
                archive.write(path, path.name)
    if output.stat().st_size >= 2**31:
        raise RuntimeError("Setup exceeds the verified 2 GiB NSIS launcher range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "web", "offline"), default="web")
    parser.add_argument("--shell", type=Path, help="Use an already assembled win-x64 directory")
    parser.add_argument("--runtime", type=Path, help="Use output from runtimework/prepare_runtime.py")
    parser.add_argument("--cache", type=Path, help="Runtime download cache")
    parser.add_argument("--work", type=Path, default=ROOT / "build")
    parser.add_argument("--out", type=Path, default=ROOT / "build/release")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2, metavar="1..8")
    parser.add_argument("--makensis", type=Path, help="Use this native Linux NSIS compiler")
    parser.add_argument("--sevenzip", type=Path, help="Use this 7-Zip instead of the one on PATH")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info < (3, 11):
        parser.error("Use Linux and Python 3.11 or newer")
    work, output_dir = args.work.resolve(), args.out.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ("web", "offline") if args.mode == "both" else (args.mode,)
    sevenzip = str(args.sevenzip.resolve()) if args.sevenzip else next(
        (path for tool in ("7zz", "7z", "7za") if (path := shutil.which(tool))), None)
    if not sevenzip:
        parser.error("A native 7-Zip with NSIS support is required. Install 7zip or pass --sevenzip.")
    nsis, env = select_nsis(str(args.makensis.resolve()) if args.makensis else None, work / ".cache/nsis")
    if not args.shell:
        subprocess.run(["bash", str(ROOT / "shellwork/build-shell.sh")], cwd=ROOT, check=True)
    shell_work = Path(os.environ.get("LTX_SHELL_WORK", ROOT / "shellwork"))
    shell = args.shell.resolve() if args.shell else shell_work.resolve() / "win-x64"
    require_files(shell, ["LTX Desktop - AMD 395 MAX.exe", "resources/app.asar", "resources/backend/ltx2_server.py",
                          "resources/comfy-engine/comfyui-v0.38.0.zip", "resources/comfy-engine/kandinsky6-1.0.2.zip",
                          "resources/comfy-engine/site-packages/ltx-comfy-deps.json"])
    runtime = args.runtime.resolve() if args.runtime else work / "runtime"
    if not args.runtime:
        cmd = [sys.executable, str(ROOT / "runtimework/prepare_runtime.py"), "--out", str(runtime),
               "--mode", "offline" if "offline" in modes else "web", "--sevenzip", sevenzip]
        if args.cache:
            cmd += ["--cache", str(args.cache.resolve())]
        subprocess.run(cmd, cwd=ROOT, check=True)
    require_files(runtime, ["install_runtime.py", "repair_single_gpu.py", "runtime-manifest.json",
                            "sitecustomize.py", "python/python.exe",
                            "vcredist/vcruntime140.dll", "vcredist/msvcp140.dll"])
    manifest_path = runtime / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["target"] != "gfx1151" or not manifest["wheels"]:
        raise ValueError("Expected a gfx1151 runtime wheel manifest")
    icon = ROOT / "app/resources/icon.ico"
    with tempfile.TemporaryDirectory(dir=work, prefix="setup-") as tmp:
        tmp = Path(tmp)
        desktop = tmp / "desktop.zip"
        build_desktop(shell, desktop)
        runtime_archive = tmp / "runtime.tar.xz"
        runtime_size = None
        if "offline" in modes:
            tree = runtime / "python-embed"
            require_files(tree, ["python.exe", "ltx-amd-runtime.json", "Lib/site-packages/torch/__init__.py"])
            marker = json.loads((tree / "ltx-amd-runtime.json").read_text(encoding="utf-8"))
            if marker["manifest_sha256"] != sha256(manifest_path) or marker["wheel_count"] != len(manifest["wheels"]):
                raise ValueError("The expanded runtime was staged from a different manifest")
            print("Compressing the offline Python runtime", flush=True)
            runtime_size = compress_runtime(tree, runtime_archive, args.threads)
        for mode in modes:
            media = [desktop, runtime_archive] if mode == "offline" else [desktop]
            stage = tmp / mode
            stage.mkdir()
            inputs, index = stage_stub_files(stage, runtime, mode, media, manifest, runtime_size)
            stub = tmp / f"{mode}-stub.exe"
            command = [nsis, "-V2", f"-DOUT={stub}", f"-DSRC={SRC}", f"-DBUNDLE={stage}", f"-DICONFILE={icon}"]
            if mode == "offline":
                command.append("-DOFFLINE")
            subprocess.run(command + [str(SRC / "setup.nsi")], env=env, check=True)
            # Publish only after checking embedded setup files, the PE header, CRC and both payload hashes.
            with tempfile.TemporaryDirectory(dir=output_dir, prefix=".staging-") as release_staging:
                artifact = Path(release_staging) / FILENAME[mode]
                seal_setup(stub, artifact, media)
                verify_installer(artifact, stub, sevenzip, inputs, index)
                checksum = sha256(artifact)
                sha_path = artifact.with_suffix(".exe.sha256")
                sha_path.write_text(f"{checksum}  {artifact.name}\n", encoding="ascii")
                os.replace(artifact, output_dir / artifact.name)
                os.replace(sha_path, output_dir / sha_path.name)
                print(f"Verified {output_dir / artifact.name}\nSHA256 {checksum}", flush=True)


if __name__ == "__main__":
    main()
