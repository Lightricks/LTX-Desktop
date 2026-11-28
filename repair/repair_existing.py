"""Apply the single-GPU runtime repair to a managed, per-user Windows installation."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ORIGINAL_DIAGNOSTIC = "403f5982d83d4147f4e23a11762e35a06e98a30ffcd1e7c579e52567728f827b"
MARKER = ".ltx-amd-managed-v1"
PRODUCT = "LTX Desktop - AMD 395 MAX"
EXECUTABLES = (f"{PRODUCT}.exe", "LTX Desktop AMD.exe")
PRIOR_RELEASE = "1.3.2-amd.1"
RELEASE = "1.3.2-amd.2"


def check_install(application):
    python = application / "resources/python/python.exe"
    markers = (MARKER, "resources/backend/ltx2_server.py",
               "resources/python/ltx-amd-runtime.json")
    if not (all((application / name).is_file() for name in markers)
            and any((application / name).is_file() for name in EXECUTABLES)):
        raise RuntimeError(f"Could not recognize the managed {PRODUCT} installation at {application}.")
    version = (application / MARKER).read_text(encoding="utf-8").strip()
    if version not in (PRIOR_RELEASE, RELEASE):
        raise RuntimeError(f"Unsupported installed release '{version}'. This repair supports {PRIOR_RELEASE} and {RELEASE}.")
    if not python.samefile(sys.executable):
        raise RuntimeError("Run Repair-LTX-Desktop-AMD.cmd so the repair uses the app's own Python.")
    return python


def require_app_closed(python):
    import psutil  # Present in the embedded runtime and independent of torch.
    app_names = {name.casefold() for name in EXECUTABLES}
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        info = proc.info
        if info["pid"] == os.getpid():
            continue
        name = (info["name"] or "").casefold()
        if name in app_names or (name == "python.exe" and info["exe"] and Path(info["exe"]) == python):
            raise RuntimeError(
                f"Close {PRODUCT} and its backend, then run the repair again. "
                f"Task Manager still shows {info['name']} (PID {info['pid']})."
            )


@contextmanager
def install_lock(state_dir):
    # Share the Windows installer's lock, including its one-byte locking region.
    import msvcrt
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "install.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another repair or installer is open. Close it and retry.") from exc
        yield


def run_logged(args):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            logging.info("%s", line.rstrip())
        return proc.wait()


def atomic_write(dest, content):
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".ltx-repair-", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_bytes(content)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def refresh_diagnostic(application, replacement):
    dest = application / "diagnose.py"
    content = replacement.read_bytes()
    if dest.is_file():
        installed = dest.read_bytes()
        if installed == content:
            return
        if hashlib.sha256(installed.replace(b"\r\n", b"\n")).hexdigest() != ORIGINAL_DIAGNOSTIC:
            logging.info("Keeping the customized installed diagnostic. The repair will run the bundled check.")
            return
    atomic_write(dest, content)


def record_release(application):
    """Keep Windows' installed-apps display synchronized with a successful repair."""
    import winreg
    marker = application / MARKER
    if marker.read_text(encoding="utf-8").strip() == PRIOR_RELEASE:
        atomic_write(marker, RELEASE.encode("utf-8"))
    registry_key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\LTXDesktopAMD"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_key, 0, winreg.KEY_READ |
                            winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as key:
            version, _ = winreg.QueryValueEx(key, "DisplayVersion")
            uninstaller, _ = winreg.QueryValueEx(key, "UninstallString")
            if version == PRIOR_RELEASE and uninstaller.strip('"').casefold() == str(
                    application.parent / "Uninstall.exe").casefold():
                winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, RELEASE)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logging.warning("Windows could not update the installed-apps version: %s", exc)


def main():
    if os.name != "nt":
        sys.exit("This repair runs under Windows. Open Repair-LTX-Desktop-AMD.cmd on the installed PC.")
    local = Path(os.environ["LOCALAPPDATA"])
    application = local / "Programs/LTXDesktopAMD/application"
    logs = local / "LTXDesktopAMD/logs"
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / "backend-repair.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, mode="a", encoding="utf-8")])
    logging.info("\n%s backend repair | %s", PRODUCT, datetime.now().isoformat(timespec="seconds"))
    logging.info("Installation: %s\nRepair log: %s", application, logfile)
    try:
        with install_lock(local / "LTXDesktopAMD/setup"):
            python = check_install(application)
            require_app_closed(python)
            logging.info("Checking the installed packages and applying the single-GPU fix...")
            cmd = [str(python), "-u", "-X", "utf8"]
            if run_logged([*cmd, str(HERE / "repair_single_gpu.py"), "--python-dir", str(python.parent)]):
                raise RuntimeError("Runtime repair stopped. See the package details above.")
            logging.info("Checking backend startup on your Radeon. The first import can take a few minutes.")
            diagnostic = HERE / "diagnose.py"
            refresh_diagnostic(application, diagnostic)
            if run_logged([*cmd, str(diagnostic), "--application", str(application)]):
                raise RuntimeError("The repair was applied, but the startup check failed. See the first traceback above.")
            record_release(application)
    except Exception as exc:
        logging.error("STOPPED: %s\nLog: %s", exc, logfile)
        return 1
    logging.info("PASS: Backend startup checked. You can now open %s from the Start menu.", PRODUCT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
