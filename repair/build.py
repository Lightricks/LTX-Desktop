"""Build the small Windows in-place repair ZIP from this checkout."""
from argparse import ArgumentParser
from pathlib import Path
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILENAME = "LTX-Desktop-AMD-Backend-Repair.zip"


def build(output):
    sources = [HERE / "Repair-LTX-Desktop-AMD.cmd", HERE / "Read-Me.txt", HERE / "repair_existing.py",
               ROOT / "runtimework/repair_single_gpu.py", HERE / "diagnose.py"]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        package = Path(tmp) / FILENAME
        with zipfile.ZipFile(package, "w") as archive:
            for source in sources:
                entry = zipfile.ZipInfo(source.name, (2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, source.read_bytes())
        with zipfile.ZipFile(package) as archive:
            if archive.testzip() or any(archive.read(p.name) != p.read_bytes() for p in sources):
                raise ValueError("Packaged content differs from source")
        package.replace(output / FILENAME)
    return output / FILENAME


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/release")
    print(build(parser.parse_args().out.resolve()))
