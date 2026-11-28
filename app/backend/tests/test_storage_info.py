"""Checkpoint-volume drive-type probe: parsers, platform dispatch, soft failure."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import services.storage_info.storage_info_impl as storage_info_impl
from services.storage_info.storage_info_impl import StorageInfoImpl
from tests.fakes.fake_storage_info import FakeStorageInfo

# Same shape the capture harness recorded (BusType from PowerShell); FriendlyName
# keeps its leading space, as NVMe namespaces actually render it.
_NVME_9100_PRO_JSON = '{"BusType":"NVMe","MediaType":"SSD","FriendlyName":" Samsung 9100 PRO 8TB"}'
_SATA_SSD_JSON = '{"BusType":"SATA","MediaType":"SSD","FriendlyName":" Samsung SSD 870 EVO 1TB"}'
_SATA_HDD_JSON = '{"BusType":"SATA","MediaType":"HDD","FriendlyName":" WDC WD40EZRZ-00GXCB0"}'


def test_parse_nvme_bus_wins_over_media_labels() -> None:
    assert storage_info_impl._parse_powershell_drive_type(_NVME_9100_PRO_JSON) == "nvme"
    assert (
        storage_info_impl._parse_powershell_drive_type(
            '{"BusType":"nvme","MediaType":"Unspecified","FriendlyName":"x"}'
        )
        == "nvme"
    )


def test_parse_sata_media_maps_to_ssd_or_hdd() -> None:
    assert storage_info_impl._parse_powershell_drive_type(_SATA_SSD_JSON) == "ssd"
    assert storage_info_impl._parse_powershell_drive_type(_SATA_HDD_JSON) == "hdd"


def test_parse_unknown_or_garbage_is_none() -> None:
    parse = storage_info_impl._parse_powershell_drive_type
    assert parse('{"BusType":"Unknown","MediaType":"Unspecified","FriendlyName":"x"}') is None
    assert parse('{"BusType":"SAS","MediaType":"SCM","FriendlyName":"x"}') is None
    assert parse('{"unrelated": true}') is None
    assert parse("null") is None
    assert parse("") is None
    assert parse("not json at all {{{") is None
    assert parse("42") is None


def test_parse_single_element_array_classifies_but_multi_is_ambiguous() -> None:
    parse = storage_info_impl._parse_powershell_drive_type
    assert parse(f"[{_SATA_SSD_JSON}]") == "ssd"
    # A spanned volume across disks is genuinely unanswerable: which disk backs reads?
    assert parse(f"[{_SATA_SSD_JSON},{_SATA_HDD_JSON}]") is None


def test_powershell_argv_pins_noninteractive_json_query() -> None:
    argv = storage_info_impl._powershell_argv_for_drive("C")
    assert argv[:4] == ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    assert "Get-Partition -DriveLetter C" in argv[4]
    assert "Get-PhysicalDisk" in argv[4]
    assert "ConvertTo-Json" in argv[4]


def test_drive_letter_extraction_is_platform_independent() -> None:
    letter = storage_info_impl._drive_letter_for_windows_path
    assert letter(Path("C:\\Models\\ckpt")) == "C"
    assert letter(Path("C:/Models/ckpt")) == "C"
    assert letter(Path("d:\\x")) == "D"
    assert letter(Path("/mnt/data/ckpt")) is None
    assert letter(Path("\\\\nas\\share\\ckpt")) is None
    assert letter(Path("relative\\ckpt")) is None


def _patched_run(monkeypatch, **behaviour: Any) -> list[list[str]]:
    """Patch subprocess.run module-wide (restored post-test); record argv."""
    calls: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        exc = behaviour.get("exc")
        if exc is not None:
            raise exc
        return SimpleNamespace(
            returncode=behaviour.get("returncode", 0),
            stdout=behaviour.get("stdout", ""),
            stderr=behaviour.get("stderr", ""),
        )

    monkeypatch.setattr(storage_info_impl.subprocess, "run", _fake_run)
    return calls


def test_run_powershell_returns_stdout_on_success(monkeypatch) -> None:
    calls = _patched_run(monkeypatch, stdout=_NVME_9100_PRO_JSON)
    out = storage_info_impl._run_powershell(["powershell", "-Command", "Get-Partition -DriveLetter C"])
    assert out == _NVME_9100_PRO_JSON
    assert calls[0][0] == "powershell"


def test_run_powershell_is_none_on_timeout_failure_or_missing_binary(monkeypatch) -> None:
    _patched_run(monkeypatch, exc=subprocess.TimeoutExpired(cmd="powershell", timeout=10))
    assert storage_info_impl._run_powershell(["powershell"]) is None

    _patched_run(monkeypatch, exc=OSError("no such binary"))
    assert storage_info_impl._run_powershell(["powershell"]) is None


def test_run_powershell_is_none_on_nonzero_exit(monkeypatch) -> None:
    _patched_run(monkeypatch, returncode=1, stderr="Get-Partition : No MSFT_Partition objects found.")
    assert storage_info_impl._run_powershell(["powershell"]) is None


def test_windows_drive_type_flows_letter_through_parse(monkeypatch) -> None:
    monkeypatch.setattr(
        storage_info_impl, "_run_powershell", lambda argv: _NVME_9100_PRO_JSON
    )
    assert storage_info_impl._windows_drive_type_for_path(Path("C:\\Models")) == "nvme"


def test_windows_drive_type_is_none_without_letter_or_output(monkeypatch) -> None:
    def _boom(argv: list[str]) -> str:
        raise AssertionError("must not invoke powershell without a drive letter")

    monkeypatch.setattr(storage_info_impl, "_run_powershell", _boom)
    assert storage_info_impl._windows_drive_type_for_path(Path("/mnt/data")) is None

    monkeypatch.setattr(storage_info_impl, "_run_powershell", lambda argv: None)
    assert storage_info_impl._windows_drive_type_for_path(Path("C:\\Models")) is None


def test_block_device_parsing() -> None:
    target = storage_info_impl._block_device_for_sysfs_target
    assert target("../../block/nvme0n1/nvme0n1p1") == "nvme0n1"
    assert target("../../block/sda/sda2") == "sda"
    assert target("../../devices/virtual/block/dm-0") == "dm-0"
    assert target("/sys/devices/virtual/block/md0") == "md0"
    assert target("garbage-without-block-segment") is None
    assert target("../../block/") is None


def test_classify_drive() -> None:
    classify = storage_info_impl._classify_drive
    assert classify(names=["nvme0n1"], rotational="0") == "nvme"
    assert classify(names=["dm-0", "nvme0n1"], rotational="0") == "nvme"
    assert classify(names=["sda"], rotational="0") == "ssd"
    assert classify(names=["mmcblk0"], rotational="0") == "ssd"
    assert classify(names=["sda"], rotational="1") == "hdd"
    assert classify(names=["sda"], rotational=None) is None
    assert classify(names=["sda"], rotational="") is None
    assert classify(names=["sda"], rotational="2") is None


def _fake_sysfs(
    tmp_path: Path,
    *,
    device: str,
    rotational: str | None,
    slaves: tuple[tuple[str, str | None], ...] = (),
    link_target: str | None = None,
) -> tuple[Path, Path]:
    """Fake /sys rooted at tmp_path plus a probe file whose st_dev selects it."""
    root = tmp_path / "sys"
    block_dir = root / "block" / device
    (block_dir / "queue").mkdir(parents=True)
    if rotational is not None:
        (block_dir / "queue" / "rotational").write_text(rotational)
    slaves_dir = block_dir / "slaves"
    slaves_dir.mkdir()
    for slave, slave_rotational in slaves:
        slave_dir = root / "block" / slave
        (slave_dir / "queue").mkdir(parents=True)
        if slave_rotational is not None:
            (slave_dir / "queue" / "rotational").write_text(slave_rotational)
        (slaves_dir / slave).symlink_to("placeholder-target")
    probe = tmp_path / "probe"
    probe.write_text("x")
    st = os.stat(probe)
    link_dir = root / "dev" / "block"
    link_dir.mkdir(parents=True)
    (link_dir / f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}").symlink_to(
        link_target if link_target is not None else f"../../block/{device}/{device}p1"
    )
    return root, probe


def test_linux_drive_type_from_fake_sysfs(tmp_path: Path) -> None:
    linux = storage_info_impl._linux_drive_type_for_path
    root, probe = _fake_sysfs(tmp_path / "nvme", device="nvme0n1", rotational="0")
    assert linux(probe, sys_root=root) == "nvme"

    root, probe = _fake_sysfs(tmp_path / "sata-ssd", device="sda", rotational="0")
    assert linux(probe, sys_root=root) == "ssd"

    root, probe = _fake_sysfs(tmp_path / "sata-hdd", device="sda", rotational="1")
    assert linux(probe, sys_root=root) == "hdd"


def test_linux_drive_type_sees_through_device_mapper(tmp_path: Path) -> None:
    root, probe = _fake_sysfs(
        tmp_path / "dm",
        device="dm-0",
        rotational="0",
        slaves=(("nvme0n1", "0"),),
        link_target="../../devices/virtual/block/dm-0",
    )
    assert storage_info_impl._linux_drive_type_for_path(probe, sys_root=root) == "nvme"


def test_linux_drive_type_falls_back_to_slave_rotational(tmp_path: Path) -> None:
    root, probe = _fake_sysfs(
        tmp_path / "dm-norot",
        device="dm-0",
        rotational=None,
        slaves=(("sda", "0"),),
        link_target="../../devices/virtual/block/dm-0",
    )
    assert storage_info_impl._linux_drive_type_for_path(probe, sys_root=root) == "ssd"


def test_linux_drive_type_is_none_when_sysfs_is_absent_or_garbage(tmp_path: Path) -> None:
    linux = storage_info_impl._linux_drive_type_for_path
    # CI containers (overlayfs): no /sys/dev/block node for the volume.
    assert linux(tmp_path, sys_root=tmp_path / "empty-sys") is None
    # Unshaped readlink target.
    root, probe = _fake_sysfs(
        tmp_path / "weird", device="sda", rotational="0", link_target="not-a-sysfs-path"
    )
    assert linux(probe, sys_root=root) is None
    # Garbage rotational content, no slaves to rescue it.
    root, probe = _fake_sysfs(tmp_path / "garbage", device="sda", rotational="banana")
    assert linux(probe, sys_root=root) is None
    # Missing path to stat.
    assert linux(tmp_path / "does-not-exist", sys_root=root) is None


def test_get_drive_type_dispatches_on_platform(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(storage_info_impl.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        storage_info_impl, "_windows_drive_type_for_path", lambda path: "nvme"
    )
    assert StorageInfoImpl(tmp_path).get_drive_type() == "nvme"

    monkeypatch.setattr(storage_info_impl.platform, "system", lambda: "Linux")
    monkeypatch.setattr(storage_info_impl, "_linux_drive_type_for_path", lambda path, **kw: "ssd")
    assert StorageInfoImpl(tmp_path).get_drive_type() == "ssd"


def test_get_drive_type_is_none_on_unimplemented_platforms(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(storage_info_impl.platform, "system", lambda: "Darwin")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not probe on Darwin")

    monkeypatch.setattr(storage_info_impl, "_windows_drive_type_for_path", _boom)
    monkeypatch.setattr(storage_info_impl, "_linux_drive_type_for_path", _boom)
    assert StorageInfoImpl(tmp_path).get_drive_type() is None


def test_get_drive_type_never_raises_on_this_machine(tmp_path: Path) -> None:
    assert StorageInfoImpl(tmp_path).get_drive_type() in (None, "nvme", "ssd", "hdd")


def test_windows_probe_failure_warns(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(storage_info_impl.platform, "system", lambda: "Windows")
    monkeypatch.setattr(storage_info_impl, "_windows_drive_type_for_path", lambda path: None)
    with caplog.at_level(logging.WARNING, logger="services.storage_info.storage_info_impl"):
        assert StorageInfoImpl(tmp_path).get_drive_type() is None
    assert any(
        record.levelname == "WARNING" and "non-NVMe" in record.getMessage()
        for record in caplog.records
    )


def test_non_windows_probe_failure_stays_debug(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(storage_info_impl.platform, "system", lambda: "Linux")
    monkeypatch.setattr(storage_info_impl, "_linux_drive_type_for_path", lambda path, **kw: None)
    with caplog.at_level(logging.WARNING, logger="services.storage_info.storage_info_impl"):
        assert StorageInfoImpl(tmp_path).get_drive_type() is None
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


def test_fake_storage_info_defaults_to_unknown() -> None:
    assert FakeStorageInfo().get_drive_type() is None
    assert FakeStorageInfo("nvme").get_drive_type() == "nvme"
