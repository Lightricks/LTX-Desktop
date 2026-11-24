"""Select an x64-capable NSIS, or install the pinned Debian package privately."""
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

from verify import sha256

PACKAGES = {
    "nsis-common_3.11-1_all.deb": "103a3284c1a5356efa0aba90fdfc391c3cde8e7df8726377cef0f98471584047",
    "nsis_3.11-1_amd64.deb": "e6eb6ffb70f0df09e95dedb2d2cf55ad3c2679f7c2ae357d0c00cc4c83178690",
}


def works(program, env):
    # -VERSION alone works on old 32-bit-only distributions.
    probe = 'Unicode true\nTarget amd64-unicode\nOutFile "probe.exe"\nSection\nnsExec::ExecToLog "cmd /c ver"\nSectionEnd\n'
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run([program, "-V1", "-"], input=probe, text=True, cwd=scratch, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode == 0, result.stdout


def select_nsis(explicit, cache):
    env = os.environ.copy()
    system = explicit or shutil.which("makensis")
    if system:
        ok, details = works(system, env)
        if ok:
            return system, env
        if explicit:
            raise RuntimeError(f"Specified makensis cannot build amd64-unicode:\n{details}")
    if platform.machine() not in ("x86_64", "amd64") or not shutil.which("dpkg-deb"):
        raise RuntimeError("Install NSIS with the amd64-unicode stubs and nsExec plugin, or pass --makensis.")
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / "root"
    for name, expected in PACKAGES.items():
        path = cache / name
        if not path.is_file() or sha256(path) != expected:
            url = "https://deb.debian.org/debian/pool/main/n/nsis/" + name
            partial = path.with_suffix(".partial")
            with urlopen(url, timeout=120) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            if sha256(partial) != expected:
                raise ValueError(f"Checksum mismatch: {url}")
            partial.replace(path)
        subprocess.run(["dpkg-deb", "-x", str(path), str(root)], check=True)
    local = str(root / "usr/bin/makensis")
    env["NSISDIR"] = str(root / "usr/share/nsis")
    ok, details = works(local, env)
    if not ok:
        raise RuntimeError(f"The private NSIS compiler cannot build x64 installers:\n{details}")
    return local, env
