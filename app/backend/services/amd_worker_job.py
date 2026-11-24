"""Keep the private worker and its descendants attached to the Desktop backend on Windows."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable


def bind_worker_to_parent(
    process: subprocess.Popen[bytes],
) -> Callable[[], None] | None:
    if os.name != "nt":
        return None
    import ctypes as c
    from ctypes import wintypes as w

    class BasicLimits(c.Structure):
        _fields_ = [
            ("per_process", c.c_int64),
            ("per_job", c.c_int64),
            ("flags", w.DWORD),
            ("min_ws", c.c_size_t),
            ("max_ws", c.c_size_t),
            ("max_processes", w.DWORD),
            ("affinity", c.c_size_t),
            ("priority", w.DWORD),
            ("scheduling", w.DWORD),
        ]

    class ExtendedLimits(c.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io_counters", c.c_uint64 * 6),
            ("proc_mem", c.c_size_t),
            ("job_mem", c.c_size_t),
            ("peak_proc", c.c_size_t),
            ("peak_job", c.c_size_t),
        ]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise c.WinError(c.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            job, 9, c.byref(limits), c.sizeof(limits)
        ):
            raise c.WinError(c.get_last_error())
        child = kernel.OpenProcess(
            0x100 | 0x1, False, process.pid
        )  # SET_QUOTA | TERMINATE
        if not child:
            raise c.WinError(c.get_last_error())
        try:
            if not kernel.AssignProcessToJobObject(job, child):
                raise c.WinError(c.get_last_error())
        finally:
            kernel.CloseHandle(child)
    except BaseException:
        kernel.CloseHandle(job)
        raise

    def release() -> None:
        kernel.CloseHandle(job)

    return release
