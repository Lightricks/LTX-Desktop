"""Compare the installed Electron frameworks and helper signatures with the pinned Mac release."""

import argparse
import hashlib
import os
from pathlib import Path
import plistlib
import stat
from zipfile import ZipFile


def verify(app: Path, original_zip: Path) -> tuple[int, int, int]:
    contents = app / "Contents"
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    prefix = "LTX Desktop.app/Contents/"
    files = links = 0
    with ZipFile(original_zip) as archive:
        original = plistlib.loads(archive.read(prefix + "Info.plist"))
        # Electron v41 resolves renamed helpers from CFBundleName. On macOS the
        # visible community name and separate bundle identifier are independent.
        if (info["CFBundleName"] != original["CFBundleName"] or
                info["CFBundleDisplayName"] != "LTX Desktop - SpaceSpeak" or
                info["CFBundleIdentifier"] != "com.spacespeak.ltx-desktop-community"):
            raise ValueError("Incorrect app identity or Electron helper lookup name")
        expected = plistlib.loads(archive.read(prefix + "_CodeSignature/CodeResources"))
        nested = [path for path, seal in expected["files2"].items()
                  if path.startswith("Frameworks/") and "cdhash" in seal]
        if not nested or "Frameworks/Electron Framework.framework" not in nested:
            raise ValueError("The original Mac release is missing its signed Electron framework")
        for name in nested:
            path = contents / name
            version = path / "Versions/Current" if path.suffix == ".framework" else path / "Contents"
            if not (version / "_CodeSignature/CodeResources").is_file():
                raise ValueError(f"The installed bundle is missing its resource envelope: {name}")
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(prefix + "Frameworks/"):
                continue
            target = contents / entry.filename[len(prefix):]
            if stat.S_ISLNK(entry.external_attr >> 16):
                if not target.is_symlink() or os.fsencode(os.readlink(target)) != archive.read(entry):
                    raise ValueError(f"Framework symlink differs from the signed release: {target}")
                if not target.exists():
                    raise ValueError(f"Broken framework symlink: {target}")
                links += 1
            else:
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"Missing installed framework file: {target}")
                with archive.open(entry) as source, target.open("rb") as actual:
                    if hashlib.file_digest(source, "sha256").digest() != hashlib.file_digest(actual, "sha256").digest():
                        raise ValueError(f"Installed framework file differs from the signed release: {target}")
                if entry.external_attr >> 16 & 0o111 and not os.access(target, os.X_OK):
                    raise ValueError(f"Installed framework executable lost its execute permission: {target}")
                files += 1
    return files, links, len(nested)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("original_zip", type=Path)
    args = parser.parse_args()
    files, links, bundles = verify(args.app, args.original_zip)
    print(f"Verified {bundles} sealed nested bundles, {files} files and {links} framework symlinks.")
