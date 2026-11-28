"""Provision the two verified parts of Lightricks' official Windows Python/CUDA runtime."""
import gzip
import io
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tarfile


class ReleaseParts(io.RawIOBase):
    """Read the release parts in order without writing a second multi-GB archive."""

    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        if not buffer:
            return 0
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract(parts, python_dir, archive_root):
    """Extract only the embedded interpreter tree from the authenticated tar stream."""
    python_dir.mkdir()
    with ReleaseParts(parts) as joined, io.BufferedReader(joined, 1024**2) as buffered:
        with gzip.GzipFile(fileobj=buffered) as gz, tarfile.open(fileobj=gz, mode="r|", bufsize=1024**2) as archive:
            for item in archive:
                name = PurePosixPath(item.name)
                if name == PurePosixPath(".") and item.isdir():
                    continue
                if (not name.parts or name.parts[0] != archive_root or ".." in name.parts
                        or ":" in item.name or "\\" in item.name or PureWindowsPath(item.name).anchor):
                    raise ValueError(f"Unexpected path in the CUDA archive: {item.name}")
                dest = python_dir.joinpath(*name.parts[1:])
                if item.isdir():
                    dest.mkdir(exist_ok=True, parents=True)
                elif item.isfile():
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    with archive.extractfile(item) as source, dest.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024**2)
                else:
                    raise ValueError(f"Unexpected entry in the Windows CUDA archive: {item.name}")
            # The end-of-tar marker can precede gzip's checksum/trailer.
            while gz.read(1024**2):
                pass
    if not (python_dir / "python.exe").is_file():
        raise ValueError("The CUDA archive is missing its Python interpreter")


def install(manifest, cache, python_dir, fetch):
    paths = []
    cache.mkdir(exist_ok=True, parents=True)
    for item in manifest["parts"]:
        name = item["filename"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid filename in CUDA runtime manifest")
        path = cache / name
        logging.info("Fetching or verifying CUDA runtime part %s (%.2f GB)", name, item["size"] / 1e9)
        fetch(item, path)
        paths.append(path)
    logging.info("Extracting the verified upstream Windows runtime.")
    extract(paths, python_dir, manifest["archiveRoot"])
    return paths


def check_python(python_dir, selection, env):
    probe = (
        "import ctypes;ctypes.windll.kernel32.SetDllDirectoryW('');"
        "import torch, fastapi, uvicorn;"
        "assert torch.version.cuda and not torch.version.hip, 'Expected a CUDA build of torch';"
    )
    if selection["backend"] == "cuda":
        probe += (
            "assert torch.cuda.is_available(), 'CUDA driver is unavailable';"
            "assert torch.ones(1,device='cuda').item()==1,'CUDA compute check failed';"
            "print(torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"
        )
    result = subprocess.run([str(python_dir / "python.exe"), "-I", "-c", probe],
                            cwd=python_dir, env=env, capture_output=True, text=True,
                            errors="replace", timeout=120)
    if result.returncode:
        raise RuntimeError(f"Upstream CUDA runtime check failed: {result.stderr.strip()}")
    if result.stdout.strip():
        logging.info("Verified %s", result.stdout.strip())
