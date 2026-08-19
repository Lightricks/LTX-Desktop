"""Windows DLL search-path hardening."""

from __future__ import annotations

import sys

import pytest

from server_utils.win_dll_search import remove_cwd_from_dll_search_path


def test_remove_cwd_from_dll_search_path_does_not_raise() -> None:
    remove_cwd_from_dll_search_path()


@pytest.mark.skipif(sys.platform != "win32", reason="SetDllDirectoryW is Windows-only")
def test_remove_cwd_clears_dll_directory_on_windows() -> None:
    import ctypes

    remove_cwd_from_dll_search_path()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetDllDirectoryW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p]
    kernel32.GetDllDirectoryW.restype = ctypes.c_uint
    buf = ctypes.create_unicode_buffer(1024)
    kernel32.GetDllDirectoryW(1024, buf)
    assert buf.value == ""
