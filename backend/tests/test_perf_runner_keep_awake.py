"""Sleep prevention during a sweep: the right OS call per platform, released at exit."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import pytest

_PERF = Path(__file__).resolve().parents[1] / "performance_runner"
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

import keep_awake  # noqa: E402


@pytest.fixture
def exit_hooks(monkeypatch) -> list:
    hooks: list = []
    monkeypatch.setattr(keep_awake.atexit, "register", hooks.append)
    return hooks


class _Process:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


def test_macos_runs_caffeinate_tied_to_this_process_and_stops_it_at_exit(monkeypatch, exit_hooks) -> None:
    started: list[list[str]] = []
    proc = _Process()

    def popen(cmd):
        started.append(cmd)
        return proc

    monkeypatch.setattr(keep_awake.sys, "platform", "darwin")
    monkeypatch.setattr(keep_awake.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(keep_awake.subprocess, "Popen", popen)

    keep_awake.start()

    assert started == [["caffeinate", "-d", "-i", "-w", str(os.getpid())]]  # -w: it dies with us even if we are killed
    assert not proc.terminated
    exit_hooks[0]()
    assert proc.terminated


def test_windows_holds_the_system_awake_and_clears_it_at_exit(monkeypatch, exit_hooks) -> None:
    calls: list[int] = []

    class _Kernel32:
        @staticmethod
        def SetThreadExecutionState(flags: int) -> int:
            calls.append(flags)
            return 1

    class _Windll:
        kernel32 = _Kernel32

    monkeypatch.setattr(keep_awake.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _Windll, raising=False)

    keep_awake.start()

    assert calls == [0x80000000 | 0x00000001 | 0x00000002]  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    exit_hooks[0]()
    assert calls[-1] == 0x80000000  # back to normal power management


def test_other_platforms_do_nothing(monkeypatch, exit_hooks) -> None:
    monkeypatch.setattr(keep_awake.sys, "platform", "linux")
    keep_awake.start()
    assert exit_hooks == []


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_failing_to_hold_the_machine_awake_warns_and_lets_the_run_continue(
    monkeypatch, exit_hooks, capsys, platform
) -> None:
    monkeypatch.setattr(keep_awake.sys, "platform", platform)
    monkeypatch.setattr(keep_awake.shutil, "which", lambda name: None)  # no caffeinate
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": type("K", (), {
        "SetThreadExecutionState": staticmethod(lambda flags: 0)})}), raising=False)

    keep_awake.start()

    assert exit_hooks == []
    assert "could not keep the machine awake" in capsys.readouterr().err
