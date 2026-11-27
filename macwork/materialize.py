"""Build the community resources from a verified Mac release and atomically install the finished app."""
import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
from zipfile import ZipFile

PRODUCT = "LTX Desktop - SpaceSpeak"
IDENTIFIER = "com.spacespeak.ltx-desktop-community"


def child(root, name):
    parts = name.split("/")
    if any(part in ("", ".", "..") or "\\" in part for part in parts):
        raise ValueError(f"Invalid resource name: {name}")
    return root.joinpath(*parts)


def assemble(payload, donor, bundle):
    contents = bundle / "Contents"
    target = contents / "Resources"
    with ZipFile(payload) as packed, (donor / "app.asar").open("rb") as source:
        plan = json.loads(packed.read("plan.json"))

        def copy(key, destination):
            size, kind, locator = plan["blobs"][key]
            if kind == "span":
                source.seek(locator)
                reader = contextlib.nullcontext(source)
            elif kind == "donor":
                reader = child(donor, locator).open("rb")
            elif kind == "payload" and key == locator:
                reader = packed.open("blobs/" + key)
            else:
                raise ValueError("Unknown resource in the Mac package.")
            checksum = hashlib.sha256()
            with reader as stream:
                remaining = size
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("Truncated Mac resource: " + key)
                    destination.write(chunk)
                    checksum.update(chunk)
                    remaining -= len(chunk)
            if checksum.hexdigest() != key:
                raise ValueError("Incorrect Mac resource checksum: " + key)

        with (target / "app.asar").open("wb") as stream:
            for key in plan["parts"]:
                copy(key, stream)
        for name, key in plan["files"]:
            path = child(target, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                copy(key, stream)

    # Stock Electron finds its original, unmodified signed helper executables in Frameworks.
    (target / "default_app.asar").unlink()
    (contents / "MacOS/Electron").rename(contents / "MacOS" / PRODUCT)
    shutil.copytree(donor / "koffi", target / "koffi", symlinks=True)
    shutil.copy2(donor / "python-deps-hash.txt", target / "python-deps-hash.txt")
    info_path = contents / "Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    info.update(CFBundleName=PRODUCT, CFBundleDisplayName=PRODUCT, CFBundleIdentifier=IDENTIFIER,
                CFBundleExecutable=PRODUCT, CFBundleIconFile="icon.icns",
                CFBundleShortVersionString=plan["version"].split("-")[0],
                CFBundleVersion=plan["version"].split("-")[0], SpaceSpeakCommunityVersion=plan["version"],
                LSMinimumSystemVersion="14.0", LSApplicationCategoryType="public.app-category.video",
                NSMicrophoneUsageDescription="LTX Desktop uses the microphone for Audio to Video.",
                ElectronAsarIntegrity={"Resources/app.asar": {"algorithm": "SHA256", "hash": plan["headerHash"]}})
    info.pop("CFBundleURLTypes", None)
    info_path.write_bytes(plistlib.dumps(info))


def activate(bundle, folder):
    target = folder / (PRODUCT + ".app")
    # Other installers can download and stage in parallel. Serialize only activation.
    with (folder / ".spacespeak-ltx-install.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.is_symlink():
            raise RuntimeError(f"Choose another applications folder; {target} is a symbolic link.")
        if target.exists():
            existing = plistlib.loads((target / "Contents/Info.plist").read_bytes())
            if existing.get("CFBundleIdentifier") != IDENTIFIER or not existing.get("SpaceSpeakCommunityVersion"):
                raise RuntimeError(f"{target} contains a different app; it was kept.")
            # RENAME_SWAP is atomic on the same macOS volume. Even a crash during
            # upgrade leaves a complete app installed. The old copy goes to staging.
            rename_swap = ctypes.CDLL(None, use_errno=True).renamex_np
            rename_swap.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename_swap.restype = ctypes.c_int
            if rename_swap(os.fsencode(bundle), os.fsencode(target), 2) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), target)
        else:
            bundle.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    build = commands.add_parser("assemble")
    build.add_argument("payload", type=Path)
    build.add_argument("donor", type=Path)
    build.add_argument("bundle", type=Path)
    install = commands.add_parser("activate")
    install.add_argument("bundle", type=Path)
    install.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.cmd == "assemble":
        assemble(args.payload, args.donor, args.bundle)
    else:
        activate(args.bundle, args.folder)
