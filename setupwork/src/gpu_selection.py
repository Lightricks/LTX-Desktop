"""Pick the Windows runtime before downloading torch. All probes use system drivers or Windows APIs."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


def _cuda_adapters():
    """Enumerate compute devices instead of the display driving the desktop (which may be Intel)."""
    try:
        driver = ctypes.WinDLL("nvcuda.dll", winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
    except OSError:
        return []

    count, version = ctypes.c_int(), ctypes.c_int()
    if (driver.cuInit(0) or driver.cuDriverGetVersion(ctypes.byref(version)) or version.value < 12000
            or driver.cuDeviceGetCount(ctypes.byref(count))):
        return []
    adapters = []
    for index in range(count.value):
        device = ctypes.c_int()
        major, minor = ctypes.c_int(), ctypes.c_int()
        size = ctypes.c_size_t()
        name = ctypes.create_string_buffer(256)
        identity = (ctypes.c_ubyte * 16)()
        if (driver.cuDeviceGet(ctypes.byref(device), index)
                or driver.cuDeviceGetName(name, len(name), device)
                or driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device)
                or driver.cuDeviceTotalMem_v2(ctypes.byref(size), device)
                or driver.cuDeviceGetUuid(ctypes.byref(identity), device)):
            continue
        adapters.append((size.value, (major.value, minor.value), name.value.decode(errors="replace"),
                         "GPU-" + str(uuid.UUID(bytes=bytes(identity)))))
    return adapters


def _windows_display_adapters():
    """PnP names can be generic on Strix Halo. Retain the PCI ID and exclude disabled adapters."""
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    query = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        "$ErrorActionPreference='Stop';"
        "$devices=@(Get-CimInstance Win32_VideoController | Where-Object {"
        "$_.ConfigManagerErrorCode -eq 0} | Select-Object Name,PNPDeviceID);"
        "ConvertTo-Json -InputObject $devices -Compress"
    )
    try:
        proc = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", query],
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                              timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
        return json.loads(proc.stdout.lstrip("\ufeff"))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Failure to enumerate adapters is distinct from a genuine empty inventory.
        raise RuntimeError(f"Windows could not enumerate display adapters. Repair WMI and rerun setup: {exc}") from exc


def select_gpu(*, exclude_cuda=False):
    # Match the v1.3.3 Windows local-generation floors in app/backend/inference/device.py.
    # CUDA-visible ordinals are not necessarily the NVML or Windows display adapter indices.
    devices = [] if exclude_cuda else _cuda_adapters()
    suitable = [(size, capability, name, identity) for size, capability, name, identity in devices
                if capability >= (7, 5) and size >= (9 if capability >= (8, 9) else 15) * 1024**3]
    if suitable:
        _, _, name, identity = max(suitable)
        return {"version": 1, "backend": "cuda", "deviceName": name, "cudaUuid": identity}
    for adapter in _windows_display_adapters():
        name = str(adapter.get("Name") or "")
        pnp = str(adapter.get("PNPDeviceID") or "").upper()
        if (re.search(r"PCI\\VEN_1002&DEV_1586(?:&|$)", pnp)
                or ("VEN_1002" in pnp and re.search(r"\b8060S\b", name, re.I))):
            return {"version": 1, "backend": "rocm", "deviceName": name or "AMD gfx1151"}
    return {"version": 1, "backend": "api", "deviceName": "No supported local GPU"}


def bind_rocm_device(python, selection):
    """Map Windows' gfx1151 PnP match to the actual HIP ordinal in the newly staged interpreter."""
    if selection["backend"] != "rocm":
        return
    script = """
import json, torch
devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    devices.append([index, getattr(props, 'gcnArchName', '').split(':')[0], props.name])
print(json.dumps(devices))
"""
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                  "GPU_DEVICE_ORDINAL"}}
    proc = subprocess.run([str(python), "-c", script], env=env, check=True, capture_output=True,
                          text=True, timeout=150)
    devices = json.loads(proc.stdout.strip().splitlines()[-1])
    for index, arch, name in devices:
        if arch == "gfx1151" or (not arch and re.search(r"\b8060S\b", name, re.I)):
            selection.update(hipOrdinal=index, deviceName=name)
            return
    raise RuntimeError(f"Windows detected gfx1151 but ROCm could not use it. Check your AMD driver. Devices: {devices!r}")


def runtime_environment(selection):
    """Child environment overrides for the installed runtime. Match electron/windows-runtime.ts."""
    backend = selection["backend"]
    if backend not in ("rocm", "cuda", "api"):
        raise ValueError(f"Unknown installed GPU backend: {backend!r}")
    env = {"LTX_GPU_RUNTIME": backend, "LTX_REQUIRE_ROCM": "1" if backend == "rocm" else "0"}
    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = selection["cudaUuid"]
    elif backend == "api":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif "hipOrdinal" in selection:
        # HIP's Windows implementation accepts HIP_VISIBLE_DEVICES; use a matching CUDA alias.
        env.update(HIP_VISIBLE_DEVICES=str(selection["hipOrdinal"]), CUDA_VISIBLE_DEVICES=str(selection["hipOrdinal"]))
    return env


def read_selection(resources):
    """Read the install-time selection. Earlier AMD-only packages had just a Python marker."""
    resources = Path(resources)
    receipt = resources / "gpu-runtime.json"
    if not receipt.exists() and (resources / "python/ltx-amd-runtime.json").is_file():
        return {"version": 1, "backend": "rocm", "deviceName": "AMD Radeon"}
    item = json.loads(receipt.read_text(encoding="utf-8"))
    if isinstance(item, dict) and item.get("version") == 1 and isinstance(item.get("deviceName"), str):
        backend = item.get("backend")
        if backend == "api":
            return item
        if backend == "rocm" and type(item.get("hipOrdinal")) is int and item["hipOrdinal"] >= 0:
            return item
        if backend == "cuda" and isinstance(item.get("cudaUuid"), str) and re.fullmatch(
            r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", item["cudaUuid"], re.I
        ):
            return item
    raise ValueError(f"Invalid installed GPU selection at {receipt}. Rerun the Windows installer.")


def child_environment(selection, environ=None):
    env = dict(os.environ if environ is None else environ)
    if selection["backend"] != "rocm" or "hipOrdinal" in selection:
        masks = {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES"}
        env = {k: v for k, v in env.items() if k.upper() not in masks}
    env.update(runtime_environment(selection))
    return env
