"""Drive-type probe for the checkpoint volume (Windows Fast DISK routing).

Disk streaming reads ~70 GB per generation: seconds on NVMe, minutes on
spinning rust. This probe classifies the volume holding the checkpoints so the
offload decision can tell them apart. Checkpoints live under the app-data tree,
so the app-data volume answers for them (volume granularity, not the file).

Every failure resolves to None: unknown means "not provably fast" downstream.
Nothing here may raise past ``get_drive_type`` — including on platforms without
an implementation (Darwin) and in containers without sysfs or PowerShell.
"""

from __future__ import annotations

import json
import logging
import ntpath
import os
import platform
import subprocess
from pathlib import Path
from typing import cast

from services.storage_info.storage_info import DriveType

logger = logging.getLogger(__name__)

# First powershell invocation pays module-load cost; past this the answer is late.
_POWERSHELL_TIMEOUT_S = 10


class StorageInfoImpl:
    """Classifies the drive holding ``checkpoint_dir``. Probe once, keep the value."""

    def __init__(self, checkpoint_dir: Path | str) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)

    def get_drive_type(self) -> DriveType | None:
        try:
            system = platform.system()
            if system == "Windows":
                result = _windows_drive_type_for_path(self._checkpoint_dir)
            elif system == "Linux":
                result = _linux_drive_type_for_path(self._checkpoint_dir)
            else:
                logger.debug("Drive-type probe: no implementation on %s", system)
                return None
            if result is None and system == "Windows":
                logger.warning(
                    "Drive-type probe failed on Windows; routing checkpoint volume as non-NVMe"
                )
            return result
        except Exception:
            logger.debug("Drive-type probe failed", exc_info=True)
            return None


def _drive_letter_for_windows_path(path: Path) -> str | None:
    """'C' for 'C:\\...' (ntpath: platform-independent); None when letterless."""
    drive, _ = ntpath.splitdrive(str(path))
    letter = drive.rstrip(":").upper()
    return letter if len(letter) == 1 and letter.isalpha() else None


def _powershell_argv_for_drive(drive_letter: str) -> list[str]:
    script = (
        f"Get-Partition -DriveLetter {drive_letter} | Get-Disk | Get-PhysicalDisk "
        "| Select-Object BusType, MediaType, FriendlyName | ConvertTo-Json -Compress"
    )
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def _run_powershell(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_POWERSHELL_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Drive-type probe: powershell invocation failed", exc_info=True)
        return None
    if proc.returncode != 0:
        logger.debug("Drive-type probe: powershell exited %s: %s", proc.returncode, proc.stderr.strip())
        return None
    return proc.stdout


def _parse_powershell_drive_type(output: str) -> DriveType | None:
    """BusType/MediaType JSON to a drive class. Only an NVMe bus claims nvme.

    Anything else degrades: a USB SSD reports SSD media and lands on ssd, never
    nvme — the fast path must be provable, not plausible.
    """
    try:
        decoded: object = json.loads(output)
    except ValueError:
        logger.debug("Drive-type probe: unparsable powershell output")
        return None
    record: object = decoded
    if isinstance(decoded, list):
        items = cast("list[object]", decoded)
        if len(items) != 1:
            logger.debug("Drive-type probe: %d disks back one volume; ambiguous", len(items))
            return None
        record = items[0]
    if not isinstance(record, dict):
        logger.debug("Drive-type probe: unexpected powershell shape")
        return None
    fields = cast("dict[str, object]", record)
    bus = fields.get("BusType")
    media = fields.get("MediaType")
    friendly = fields.get("FriendlyName")
    bus_name = bus.strip().lower() if isinstance(bus, str) else ""
    media_name = media.strip().lower() if isinstance(media, str) else ""
    friendly_name = friendly.strip() if isinstance(friendly, str) else ""
    result: DriveType | None = None
    if bus_name == "nvme":
        result = "nvme"
    elif media_name == "ssd":
        result = "ssd"
    elif media_name == "hdd":
        result = "hdd"
    logger.debug(
        "Drive-type probe: BusType=%s MediaType=%s FriendlyName=%s -> %s",
        bus_name or "?",
        media_name or "?",
        friendly_name or "?",
        result,
    )
    return result


def _windows_drive_type_for_path(path: Path) -> DriveType | None:
    letter = _drive_letter_for_windows_path(path)
    if letter is None:
        logger.debug("Drive-type probe: no drive letter in %s", path)
        return None
    output = _run_powershell(_powershell_argv_for_drive(letter))
    if output is None:
        return None
    return _parse_powershell_drive_type(output)


def _block_device_for_sysfs_target(target: str) -> str | None:
    """'nvme0n1' from a /sys/dev/block readlink target. None when unshaped."""
    parts = Path(target).parts
    try:
        rest = parts[parts.index("block") + 1 :]
    except ValueError:
        return None
    return rest[0] if rest else None


def _read_rotational(block_dir: Path) -> str | None:
    try:
        return (block_dir / "queue" / "rotational").read_text().strip()
    except OSError:
        return None


def _classify_drive(*, names: list[str], rotational: str | None) -> DriveType | None:
    if rotational not in ("0", "1"):
        return None
    if rotational == "1":
        return "hdd"
    return "nvme" if any(name.startswith("nvme") for name in names) else "ssd"


def _linux_drive_type_for_path(path: Path, *, sys_root: Path = Path("/sys")) -> DriveType | None:
    """st_dev through /sys/block: slaves rescue device-mapper stacking (dm-0)."""
    try:
        st = os.stat(path)
    except OSError:
        logger.debug("Drive-type probe: cannot stat %s", path)
        return None
    link = sys_root / "dev" / "block" / f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    try:
        target = os.readlink(link)
    except OSError:
        logger.debug("Drive-type probe: no sysfs node for %s", link)
        return None
    device = _block_device_for_sysfs_target(target)
    if device is None:
        logger.debug("Drive-type probe: unshaped sysfs target %r", target)
        return None
    block_dir = sys_root / "block" / device
    try:
        slaves = sorted(p.name for p in (block_dir / "slaves").iterdir())
    except OSError:
        slaves = []
    names = [device, *slaves]
    rotational = _read_rotational(block_dir)
    if rotational is None:
        for slave in slaves:
            rotational = _read_rotational(sys_root / "block" / slave)
            if rotational is not None:
                break
    result = _classify_drive(names=names, rotational=rotational)
    logger.debug(
        "Drive-type probe: %s names=%s rotational=%s -> %s", path, names, rotational, result
    )
    return result
