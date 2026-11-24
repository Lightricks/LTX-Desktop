"""Remove the current working directory from the Windows DLL search path."""

from __future__ import annotations

import sys


def remove_cwd_from_dll_search_path() -> None:
    """Call ``SetDllDirectoryW("")`` so ``LoadLibrary`` does not search CWD.

    No-op off Windows. Must run before importing native extensions (torch, etc.).
    """
    if sys.platform != "win32":
        return

    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
    kernel32.SetDllDirectoryW.restype = ctypes.c_bool
    if not kernel32.SetDllDirectoryW(""):
        print(f"SetDllDirectoryW failed: {ctypes.get_last_error()}", file=sys.stderr)
