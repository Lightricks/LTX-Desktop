"""Keep the inference child alive only while its owning backend pipe is open."""

import subprocess
import sys
from threading import Thread


def main() -> int:
    child = subprocess.Popen(
        sys.argv[1:], stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def watch_parent() -> None:
        # EOF also arrives if Electron force-kills or the backend crashes.
        sys.stdin.buffer.read()
        if child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            except ProcessLookupError:
                pass

    Thread(target=watch_parent, daemon=True).start()
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
