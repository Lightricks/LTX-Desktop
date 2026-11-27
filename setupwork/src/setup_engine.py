"""Per-user installer for the Windows LTX Desktop community branch."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

MARKER = ".ltx-amd-managed-v1"
RELEASE = "1.3.3-community.2"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def matches(path, item):
    return (path.is_file() and path.stat().st_size == item["size"]
            and digest(path) == item["sha256"])

def owned_remove(path):
    if path.exists():
        if path.is_symlink() or not (path / MARKER).is_file():
            raise RuntimeError(f"An unmanaged directory already exists at {path}.")
        shutil.rmtree(path)

def fetch(item, path):
    if matches(path, item):
        return
    url = item["url"]
    if not url or not url.startswith("https://"):
        raise RuntimeError(f"Missing bundled runtime component: {path.name}")
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(4):
        try:
            start = partial.stat().st_size if partial.is_file() else 0
            if start >= item["size"]:
                if matches(partial, item):
                    partial.replace(path)
                    return
                partial.unlink()
                start = 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                append = start > 0 and response.status == 206
                if response.status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise OSError("Unexpected range in download response")
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            if not matches(partial, item):
                if partial.stat().st_size >= item["size"]:
                    partial.unlink()
                raise OSError("Downloaded data failed size or SHA-256 verification")
            partial.replace(path)
            return
        except (OSError, TimeoutError) as err:
            if attempt == 3:
                raise RuntimeError(f"Could not download {path.name}: {err}") from err
            time.sleep(2 ** attempt)

def wheelhouse(items, bundle, adjacent, cache):
    # The offline distribution includes every public wheel. Local project wheels
    # are embedded in Setup, so a web installation never needs a custom server.
    if adjacent.is_dir() and all((adjacent / w["filename"]).is_file() for w in items):
        logging.info("Using the included offline runtime.")
        return adjacent
    cache.mkdir(parents=True, exist_ok=True)
    def stage(item):
        name = item["filename"]
        if not name or Path(name).name != name:
            raise ValueError("Invalid package name in installer manifest")
        dest = cache / name
        for directory in (bundle / "wheels", adjacent):
            src = directory / name
            if src.is_file() and matches(src, item):
                if not matches(dest, item):
                    shutil.copyfile(src, dest)
                return name
        fetch(item, dest)
        return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(stage, item) for item in items]
        for index, task in enumerate(concurrent.futures.as_completed(tasks), 1):
            logging.info("Verified %d/%d: %s", index, len(items), task.result())
    return cache

def exec_logged(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors="replace", encoding="utf-8") as proc:
        for line in proc.stdout:
            logging.info("%s", line.rstrip())
        if proc.wait():
            raise RuntimeError(f"Runtime setup failed (code {proc.returncode}).")

def unpack_size(archive):
    with zipfile.ZipFile(archive) as z:
        return sum(x.file_size for x in z.infolist())

def restore_assets(stage):
    index = stage / "amd-asset-offsets.json"
    assets = json.loads(index.read_text(encoding="utf-8"))
    with (stage / "resources/app.asar").open("rb") as blob:
        for asset in assets:
            dest = (stage / asset["path"]).resolve()
            if not dest.is_relative_to(stage.resolve()) or dest.exists():
                raise ValueError(f"Invalid packed asset destination: {dest}")
            blob.seek(asset["offset"])
            data = blob.read(asset["size"])
            if hashlib.sha256(data).hexdigest() != asset["sha256"]:
                raise ValueError(f"Cannot restore asset {dest.name} from application archive")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    index.unlink()

def preserve_launcher_alias(current, stage):
    # User-created shortcuts and existing taskbar pins may still target the old path.
    old_launcher = "LTX Desktop AMD.exe"
    if (current / MARKER).is_file() and (current / old_launcher).is_file():
        try:
            os.link(stage / "LTX Desktop - AMD 395 MAX.exe", stage / old_launcher)
        except OSError as exc:
            logging.warning("Could not retain the former launcher path for existing shortcuts: %s", exc)


def require_free_space(root, required):
    margin = 512 * 1024**2
    if shutil.disk_usage(root).free < required + margin:
        raise RuntimeError(f"Free at least {(required + margin) / 1024**3:.1f} GiB on the application drive.")


def install_rocm(bundle, resources, cache, adjacent):
    manifest = bundle / "runtime-manifest.json"
    items = json.loads(manifest.read_text(encoding="utf-8"))["wheels"]
    python_dir = resources / "python"
    archive = bundle / "runtime.tar.xz"
    logging.info("Installing the native gfx1151 runtime. Large DLLs may take several minutes.")
    downloads = []
    if archive.is_file():
        info = json.loads((bundle / "runtime-size.json").read_text(encoding="utf-8"))
        require_free_space(resources, info["unpacked_bytes"])
        with tarfile.open(archive, "r|xz") as payload:
            payload.extractall(python_dir, filter="data")
    else:
        packages = wheelhouse(items, bundle, adjacent, cache)
        require_free_space(resources, sum(unpack_size(packages / w["filename"]) for w in items))
        shutil.copytree(bundle / "python", python_dir)
        exec_logged([sys.executable, "-u", "-X", "utf8", str(bundle / "install_runtime.py"),
                     "--python-dir", str(python_dir), "--wheelhouse", str(packages), "--manifest", str(manifest)])
        if packages == cache:
            downloads = [cache / item["filename"] for item in items]
    exec_logged([sys.executable, "-u", "-X", "utf8", str(bundle / "repair_single_gpu.py"),
                 "--python-dir", str(python_dir)])
    shutil.copyfile(bundle / "sitecustomize.py", python_dir / "Lib/site-packages/sitecustomize.py")
    if (bundle / "vcredist").is_dir():
        shutil.copytree(bundle / "vcredist", python_dir, dirs_exist_ok=True)
    if not (python_dir / "ltx-amd-runtime.json").is_file():
        raise RuntimeError("The Radeon runtime receipt is missing.")
    shutil.copyfile(manifest, resources / manifest.name)
    return downloads


def install_runtime_for_hardware(bundle, resources, adjacent, cache, selection):
    # setup_engine is an extracted script when bootstrapped by Windows embedded Python.
    # The package form also allows repository tests to exercise this exact dispatch.
    if __package__:
        from . import cuda_runtime, gpu_selection
    else:
        import cuda_runtime, gpu_selection

    python_dir = resources / "python"
    downloads = []
    if selection["backend"] in ("cuda", "api"):
        source = bundle / "cuda-runtime.json"
        manifest = json.loads(source.read_text(encoding="utf-8"))
        part_cache = cache / ("cuda-" + manifest["release"])
        # Download and install directories are on the same per-user volume. Already
        # verified parts and resumable partials do not need a second allocation.
        remaining = 0
        for item in manifest["parts"]:
            path = part_cache / item["filename"]
            if not matches(path, item):
                partial = path.with_suffix(path.suffix + ".partial")
                saved = partial.stat().st_size if partial.is_file() else 0
                remaining += max(0, item["size"] - saved)
        require_free_space(resources, manifest["unpackedBytes"] + remaining)
        downloads = cuda_runtime.install(manifest, part_cache, python_dir, fetch)
        try:
            cuda_runtime.check_python(python_dir, selection, gpu_selection.child_environment(selection))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if selection["backend"] != "cuda":
                raise
            fallback = gpu_selection.select_gpu(exclude_cuda=True)
            if fallback["backend"] != "rocm":
                raise RuntimeError("The installed NVIDIA driver could not run the pinned CUDA runtime. "
                                   f"Update the NVIDIA driver and rerun setup. {exc}") from exc
            logging.warning("The CUDA probe failed. Switching to the detected Radeon: %s", exc)
            shutil.rmtree(python_dir)
            selection = fallback
        else:
            shutil.copyfile(source, resources / source.name)

    if selection["backend"] == "rocm":
        downloads.extend(install_rocm(bundle, resources, cache, adjacent))
        gpu_selection.bind_rocm_device(python_dir / "python.exe", selection)
    elif selection["backend"] not in ("cuda", "api"):
        raise ValueError(f"Unknown GPU target: {selection['backend']}")

    (resources / "gpu-runtime.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return downloads


def deploy(bundle, root, adjacent, cache, compact=None):
    if __package__:
        from .gpu_selection import select_gpu
    else:
        from gpu_selection import select_gpu

    if os.name == "nt":
        resolved = str(root.resolve())
        if not resolved.startswith("\\\\?\\"):
            prefix = "\\\\?\\UNC\\" if resolved.startswith("\\\\") else "\\\\?\\"
            root = Path(prefix + resolved.lstrip("\\"))
    current, stage, prior = (root / x for x in ("application", "installing", "previous"))
    if prior.exists() and not current.exists():
        if not (prior / MARKER).is_file():
            raise RuntimeError(f"Cannot recover unmanaged directory {prior}")
        prior.rename(current)
    for directory in (stage, prior):
        owned_remove(directory)
    if current.exists() and not (current / MARKER).is_file():
        raise RuntimeError(f"Cannot replace unmanaged directory {current}")
    # Only historical offline media is specific to gfx1151. The shared online
    # installer probes the host before fetching any vendor-specific runtime.
    offline = (bundle / "runtime.tar.xz").is_file()
    selection = select_gpu(exclude_cuda=offline)
    if offline and selection["backend"] != "rocm":
        raise RuntimeError("This offline media requires Radeon gfx1151. Use the community online installer.")
    logging.info("Selected %s runtime for %s.", selection["backend"], selection["deviceName"])
    if compact is None:
        required = unpack_size(bundle / "desktop.zip")
        with zipfile.ZipFile(bundle / "desktop.zip") as z:
            required += sum(asset["size"] for asset in json.loads(z.read("amd-asset-offsets.json")))
    else:
        required = compact.required_bytes(bundle / "compact")
    require_free_space(root, required)
    stage.mkdir()
    (stage / MARKER).write_text(RELEASE, encoding="utf-8")
    compact_downloads = ()
    try:
        logging.info("Extracting LTX Desktop - AMD 395 MAX.")
        if compact is None:
            with zipfile.ZipFile(bundle / "desktop.zip") as z:
                z.extractall(stage)
            restore_assets(stage)
        else:
            compact_downloads = compact.hydrate(stage, bundle / "compact", cache / "compact", fetch)
        resources = stage / "resources"
        runtime_downloads = install_runtime_for_hardware(bundle, resources, adjacent, cache, selection)
        for path in (stage / "LTX Desktop - AMD 395 MAX.exe", resources / "python/python.exe",
                     resources / "gpu-runtime.json", resources / "backend/ltx2_server.py",
                     stage / "resources/comfy-engine/comfyui-v0.38.0.zip",
                     stage / "resources/comfy-engine/kandinsky6-1.0.2.zip",
                     stage / "resources/comfy-engine/site-packages/ltx-comfy-deps.json"):
            if not path.is_file():
                raise RuntimeError(f"Required installed file missing: {path.name}")
        shutil.copyfile(bundle / "AMD-Diagnostics.cmd", stage / "AMD-Diagnostics.cmd")
        shutil.copyfile(bundle / "diagnose.py", stage / "diagnose.py")
        shutil.copyfile(bundle / "gpu_selection.py", stage / "gpu_selection.py")
        preserve_launcher_alias(current, stage)
        if current.exists():
            current.rename(prior)
        try:
            stage.rename(current)
        except Exception:
            if prior.exists():
                prior.rename(current)
            raise
    except Exception:
        owned_remove(stage)
        raise
    for path in (prior,):
        try:
            owned_remove(path)
        except OSError as err:
            logging.warning("Previous version cleanup deferred: %s", err)
    for path in runtime_downloads:
        try:
            path.unlink(missing_ok=True)
        except OSError as err:
            logging.warning("Keeping cached component %s: %s", path.name, err)
    if compact is not None:
        compact.clear_downloads(compact_downloads)
    logging.info("Installed %s", current)

def unpack_media(setup_exe, bundle):
    offline = bundle / "offline-index.json"
    online = bundle / "online-index.json"
    if offline.is_file() == online.is_file():
        raise ValueError("Expected exactly one installation media index")
    index = offline if offline.is_file() else online
    expected_names = {"desktop.zip", "runtime.tar.xz"} if index == offline else {"desktop.zip"}
    files = json.loads(index.read_text(encoding="utf-8"))
    if set(files) != expected_names:
        raise ValueError("Invalid installation media index")
    logging.info("Reading packaged installation media.")
    with zipfile.ZipFile(setup_exe) as media:
        if sorted(media.namelist()) != sorted(files):
            raise ValueError("The installer is incomplete")
        if shutil.disk_usage(bundle).free <= sum(x["size"] for x in files.values()):
            raise OSError("Insufficient free space on the Windows temporary drive")
        for name, expected in files.items():
            entry = media.getinfo(name)
            if entry.file_size != expected["size"] or entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Invalid installation payload: {name}")
            checksum = hashlib.sha256()
            with media.open(entry) as source, (bundle / name).open("wb") as dest:
                while block := source.read(2**20):
                    dest.write(block)
                    checksum.update(block)
            if checksum.hexdigest() != expected["sha256"]:
                raise ValueError(f"SHA-256 verification failed: {name}")
            logging.info("Verified %s", name)

def main():
    import msvcrt
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("installer_dir", type=Path)
    parser.add_argument("setup_exe", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle.resolve()))
    local = Path(os.environ["LOCALAPPDATA"])
    root = local / "Programs" / "LTXDesktopAMD"
    state = local / "LTXDesktopAMD" / "setup"
    state.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    logfile = state / "install.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(logfile, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    try:
        with (state / "install.lock").open("a+b") as lock:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another installation is in progress.") from exc
            compact = None
            support = args.bundle / "compact"
            if support.is_dir():
                # Windows embedded Python only searches its own zip and directory.
                sys.path.insert(0, str(support.resolve()))
                import compact_setup as compact
            else:
                unpack_media(args.setup_exe, args.bundle)
            deploy(args.bundle, root, args.installer_dir / "runtime-packages", state / "downloads", compact)
    except Exception:
        logging.exception("Installation did not finish. Close LTX Desktop - AMD 395 MAX before retrying setup.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
