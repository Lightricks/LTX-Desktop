"""Keep the machine fully active while a long sweep runs.

A regression sweep takes from half an hour to several hours. If the machine idles into
sleep the backend stalls and the run dies with a connection error halfway through. The
display is held on too, so the screen saver and the lock screen that follows it never start.

* macOS: ``caffeinate -d -i -w <our pid>`` prevents display and idle sleep and exits on
  its own when this process does, even if this process is killed.
* Windows: ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)``
  holds the system and display on until it is cleared or the process ends.
* Other platforms: nothing (no portable way without wrapping the whole process).

Closing a laptop lid still sleeps it on both platforms unless it is on power with an
external display; that is an OS rule these calls cannot override.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess
import sys
from collections.abc import Callable

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def start() -> None:
    """Hold the machine awake until this process exits. Failure only warns: a run that
    cannot prevent sleep is still worth running."""
    try:
        release = _hold_macos() if sys.platform == "darwin" else _hold_windows() if sys.platform == "win32" else None
    except OSError as exc:
        print(f"[sanity] could not keep the machine awake ({exc}); disable sleep manually for long runs",
              file=sys.stderr)
        return
    if release is None:
        return
    atexit.register(release)
    print("[sanity] keeping the machine awake while the sweep runs", flush=True)


def _hold_macos() -> Callable[[], None]:
    if shutil.which("caffeinate") is None:
        raise OSError("caffeinate not found")
    proc = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
    return proc.terminate


def _hold_windows() -> Callable[[], None]:
    if sys.platform != "win32":  # also tells the type checker that windll exists below
        raise OSError("not Windows")
    kernel32 = ctypes.windll.kernel32
    if kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED) == 0:
        raise OSError("SetThreadExecutionState failed")
    return lambda: kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
