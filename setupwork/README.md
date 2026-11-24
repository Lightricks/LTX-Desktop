# Build the Windows installers on Linux

From the repository root, run:

```sh
python3 setupwork/build.py
```

The build compiles the desktop shell, prepares the Windows runtime, and creates these files under `build/release/` with an adjacent SHA-256 file for each one.

| File | Contents |
| --- | --- |
| `LTX-Desktop-AMD-395-MAX-1.3.2-Windows-x64-Setup.exe` | Editor and expanded AMD runtime compressed into the installer |
| `LTX-Desktop-AMD-395-MAX-Online-Setup.exe` | Editor, embedded Python, and project-specific wheels. Setup downloads the remaining wheels using the pinned runtime manifest |

Model downloads happen after installation. The online installer embeds the small private wheels and the release's single-GPU compatibility patches. After verifying and unpacking the pinned public wheels, it applies the same patches used when building the offline runtime.

## Build requirements

Use a Linux host with Python 3.11 or newer, GNU tar, xz-utils, and 7-Zip with its NSIS format handler. Debian 13 provides these through `python3`, `tar`, `xz-utils`, and `7zip`. Install the shell and runtime build prerequisites in the repository build guide, including Node and Corepack. For an initial build, allow space for the working Windows runtime, downloads, and both packaged installers.

A system `makensis` on PATH is preferred. It must compile `Target amd64-unicode` with the 64-bit nsExec plug-in. Debian 13 ships a working NSIS 3.11. If the compiler is missing or cannot build an x64 stub, the builder on x86-64 Linux fetches the pinned Debian 13 `nsis` and `nsis-common` packages, checks their SHA-256 hashes, and extracts them below `build/.cache/nsis/` with `dpkg-deb`. This private compiler requires glibc 2.38 or newer, libstdc++ from GCC 13.1 or newer, libgcc, and zlib. Other distributions can supply their own working NSIS through `--makensis`. Use `--sevenzip` to supply the executable from an alternate 7-Zip installation.

## Iterating on packaging

To reuse an assembled shell and staged runtime, run:

```sh
python3 setupwork/build.py --shell shellwork/win-x64 --runtime build/runtime
```

Use `--mode web` for the smaller online installer or `--mode offline` for the standalone installer. A web runtime staging directory is sufficient for `--mode web`. Offline and combined builds require `runtimework/prepare_runtime.py --mode offline`. The offline compression defaults to two threads. Increase `--threads` if memory permits. `--out` changes the release destination. `--work` changes the temporary build and cache location.

Each build derives fresh media indexes from its own inputs. Before publishing each executable, the builder checks the AMD64 PE and NSIS CRC, reconstructs and verifies the files embedded in NSIS, and verifies the appended archive's complete file list and SHA-256 digests. For a quick test of the installer's extraction and corruption detection code, run `python3 setupwork/test_media.py`.

The finished installers install for the current Windows user. GPU execution and model generation require a compatible Windows Radeon system.
