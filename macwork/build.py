"""Build the Apple Silicon online installer using the compiled community renderer and a pinned Mac release."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
from zipfile import ZipFile, ZipInfo, ZIP_STORED, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"
DONOR_HASH = "90bad9cc1662e52f3d291ca2e44e2bee5651c5497e1720477317a6c616db7f51"
ELECTRON_LICENSE_ARCHIVE_HASH = "8e18ef53da62bca6132508721c1f94e06b5773b48d366b95e593479892f0a2fe"
DONOR_RES = "LTX Desktop.app/Contents/Resources/"
PRODUCT = "LTX Desktop - SpaceSpeak"
MAC_VERSION = "1.3.3-community.6"
BLOCK_SIZE = 4 * 1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_entry(archive, name, data, *, executable=False, compression=ZIP_DEFLATED):
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (0o100755 if executable else 0o100644) << 16
    archive.writestr(info, data, compress_type=compression)


def nodes(tree, prefix=""):
    for name, node in tree.items():
        path = prefix + name
        if "files" in node:
            yield from nodes(node["files"], path + "/")
        else:
            yield path, node


def pack_header(header):
    body = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    padded = (len(body) + 3) & ~3
    return struct.pack("<4I", 4, 8 + padded, 4 + padded, len(body)) + body.ljust(padded, b"\0")


def build(upstream_zip, electron_zip, output):
    with upstream_zip.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != DONOR_HASH:
            raise ValueError("The upstream Mac ZIP does not match Lightricks v1.3.3.")
    with electron_zip.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != ELECTRON_LICENSE_ARCHIVE_HASH:
            raise ValueError("The Electron license archive does not match Electron 41.0.2 for macOS arm64.")
    for required in ("dist/index.html", "dist-electron/main.js", "dist-electron/preload.js", "dist-remote/index.html"):
        if not (APP / required).is_file():
            raise ValueError("Build the community frontend before packaging: " + required)
    h3_shader = APP / "backend/services/h3_metal/int8_gemm.metal"
    if not h3_shader.is_file():
        raise FileNotFoundError(h3_shader)
    manifest = json.loads((APP / "package.json").read_text())
    # Electron's packaged module tree in the official distribution is for macOS arm64.
    # Preserve its metadata and ordinals while replacing every application entry point.
    with ZipFile(upstream_zip) as donor:
        packed = donor.read(DONOR_RES + "app.asar")
        _, header_size, _, json_size = struct.unpack_from("<4I", packed)
        source = json.loads(packed[16:16 + json_size])["files"]
        base = 8 + header_size
        index = {}
        for path, node in nodes(source):
            if "link" in node:
                continue
            key = node["integrity"]["hash"]
            if node.get("unpacked"):
                index[key] = ["donor", "app.asar.unpacked/" + path]
            else:
                index[key] = ["span", base + int(node["offset"])]
        # Remote and Explore media are loose files in both distributions.
        for item in donor.infolist():
            if item.is_dir() or not item.filename.startswith((DONOR_RES + "dist-remote/", DONOR_RES + "explore-assets/")):
                continue
            key = digest(donor.read(item))
            index.setdefault(key, ["donor", item.filename[len(DONOR_RES):]])

    archive_bytes = io.BytesIO()
    blobs, parts, loose = {}, [], []
    with ZipFile(archive_bytes, "w") as payload:
        def record(data):
            key = digest(data)
            if key not in blobs:
                location = index.get(key)
                if location is None:
                    write_entry(payload, "blobs/" + key, data)
                    location = ["payload", key]
                blobs[key] = [len(data), *location]
            return key

        header = {"files": {"node_modules": source["node_modules"]}}
        offset = 0
        for path, entry in nodes(header["files"]):
            if "link" in entry:
                continue
            key = entry["integrity"]["hash"]
            blobs[key] = [entry["size"], *index[key]]
            if entry.get("unpacked"):
                loose.append(["app.asar.unpacked/" + path, key])
            else:
                parts.append(key)
                entry["offset"] = str(offset)
                offset += entry["size"]

        def add_app_file(path, data, unpacked=False):
            nonlocal offset
            tree = header["files"]
            *folders, name = path.split("/")
            for folder in folders:
                tree = tree.setdefault(folder, {"files": {}})["files"]
            key = record(data)
            blocks = [digest(data[i:i + BLOCK_SIZE]) for i in range(0, len(data), BLOCK_SIZE)] or [key]
            entry = {"size": len(data), "integrity": {
                "algorithm": "SHA256", "hash": key, "blockSize": BLOCK_SIZE, "blocks": blocks}}
            if unpacked:
                entry["unpacked"] = True
                loose.append(["app.asar.unpacked/" + path, key])
            else:
                entry["offset"] = str(offset)
                parts.append(key)
                offset += len(data)
            tree[name] = entry

        for folder in ("dist", "dist-electron"):
            for path in sorted((APP / folder).rglob("*")):
                if path.is_file() and path.suffix != ".map":
                    name = path.relative_to(APP).as_posix()
                    add_app_file(name, path.read_bytes(), name.startswith("dist/splash/"))
        for name in ("LICENSE.txt", "NOTICES.md"):
            add_app_file(name, (APP / name).read_bytes())
        metadata = {key: manifest[key] for key in ("type", "main", "license", "homepage")}
        metadata.update(version=MAC_VERSION, name="ltx-desktop-spacespeak", productName=PRODUCT,
                        description="Community fork of Lightricks LTX Desktop by Peter Beery / SpaceSpeak.com")
        add_app_file("package.json", json.dumps(metadata, indent=2).encode())

        def add_resources(root, prefix, include):
            for path in sorted(root.rglob("*")):
                if path.is_file() and include(path):
                    loose.append([prefix + path.relative_to(root).as_posix(), record(path.read_bytes())])

        def backend_file(path):
            return (not set(path.relative_to(APP / "backend").parts).intersection({"tests", ".venv", "__pycache__"})
                    and (path.suffix in (".py", ".json", ".sql") or path.name == "pyproject.toml"
                         or path == h3_shader))

        add_resources(APP / "backend", "backend/", backend_file)
        add_resources(APP / "dist-remote", "dist-remote/", lambda p: p.suffix != ".map")
        add_resources(APP / "shared", "shared/", lambda p: p.suffix == ".json")
        add_resources(APP / "resources/licenses", "licenses/", lambda p: True)
        add_resources(APP / "resources/explore-assets", "explore-assets/", lambda p: True)
        # H3 uses the same pinned ComfyUI core as the Windows worker. Keep the
        # archive inside Resources; the worker unpacks it in the user's profile.
        core_name = "comfyui-v0.38.0.zip"
        relative_core = "vendor/ComfyUI/" + core_name
        assets = json.loads((HERE.parent / "source-assets.json").read_bytes())["files"]
        core_asset = next(entry for entry in assets if entry["path"] == relative_core)
        core = (HERE.parent / relative_core).read_bytes()
        if len(core) != core_asset["size"] or digest(core) != core_asset["sha256"]:
            raise ValueError("Incorrect ComfyUI archive. Run restore_sources.py before building.")
        loose.append(["comfy-engine/" + core_name, record(core)])
        loose.append(["licenses/comfyui/LICENSE", record((HERE.parent / "vendor/ComfyUI/LICENSE").read_bytes())])
        loose.append(["icon.icns", record((APP / "resources/icon.icns").read_bytes())])
        # Ship the complete notices from Electron/Chromium. The end user's Mac only
        # needs the LTX archive, which contains the already-sealed native bundles.
        with ZipFile(electron_zip) as licenses:
            for name in ("LICENSE", "LICENSES.chromium.html"):
                loose.append(["licenses/electron/" + name, record(licenses.read(name))])


        encoded_header = pack_header(header)
        parts.insert(0, record(encoded_header))
        raw_length = struct.unpack_from("<I", encoded_header, 12)[0]
        plan = {"version": MAC_VERSION, "parts": parts, "files": loose, "blobs": blobs,
                "headerHash": digest(encoded_header[16:16 + raw_length])}
        write_entry(payload, "plan.json", json.dumps(plan, separators=(",", ":")).encode())

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w") as installer:
        prefix = PRODUCT + " Setup/"
        write_entry(installer, prefix + "community.zip", archive_bytes.getvalue(), compression=ZIP_STORED)
        for name in ("materialize.py", "Install LTX Desktop.command"):
            write_entry(installer, prefix + name, (HERE / name).read_bytes(), executable=name.endswith(".command"))
        write_entry(installer, prefix + "entitlements.mac.plist", (APP / "resources/entitlements.mac.plist").read_bytes())
        write_entry(installer, prefix + "comfy-worker.mac.plist", (HERE / "comfy-worker.mac.plist").read_bytes())
        for check in ("check_native_mps.py", "check_h3_m5_kernel.py"):
            write_entry(installer, prefix + check, (HERE.parent / "checks" / check).read_bytes())
        write_entry(installer, prefix + "README.txt", (HERE / "README.txt").read_bytes())
    print(f"{output}: {output.stat().st_size:,} bytes, {len(parts)} ASAR spans and {len(loose)} resources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-zip", required=True, type=Path)
    parser.add_argument("--electron-zip", required=True, type=Path, help="Electron 41.0.2 darwin-arm64 ZIP (license notices only)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.upstream_zip, args.electron_zip, args.output)
