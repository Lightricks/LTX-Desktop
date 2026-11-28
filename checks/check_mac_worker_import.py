"""Load a native dependency from the private H3 wheel directory on macOS arm64."""
import argparse
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources", type=Path)
    resources = parser.parse_args().resources.resolve()
    if (sys.platform != "darwin" or platform.machine() != "arm64" or
            Path(sys.executable).resolve() != resources / "python/bin/ltx-comfy-python"):
        raise SystemExit("Run with the installed Mac community worker interpreter.")
    sys.path.insert(0, str(resources / "backend"))
    from services.comfy_posix_wheels import manifest_file, provision_wheels

    with TemporaryDirectory() as directory:
        temp = Path(directory)
        overlay = temp / "site"
        provision_wheels(manifest_file("darwin"), temp / "cache", overlay, None)
        sys.path.insert(0, str(overlay))
        # Exercise an external native module and an external dylib dependency tree.
        import msgpack._cmsgpack as native
        import av
        import av._core
        for module in (native, av._core):
            if not Path(module.__file__).resolve().is_relative_to(overlay):
                raise RuntimeError(f"Loaded an unexpected {module.__name__} at {module.__file__}")
        pixel = av.VideoFrame(2, 2, "rgb24")
        if pixel.reformat(format="yuv420p").format.name != "yuv420p":
            raise RuntimeError("Private PyAV/FFmpeg failed to convert a frame.")
    print("Passed native dependency imports with the installed Apple silicon worker.")


if __name__ == "__main__":
    main()
