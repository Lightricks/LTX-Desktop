# Compact Windows online installer

This optional packaging route produces a per-user Windows x64 installer for LTX Desktop - AMD 395 MAX 1.3.3-community.3. Setup chooses the pinned NVIDIA or AMD runtime for the detected hardware. It includes Python, the project-specific parts of the application and the verification data. Setup retrieves the other components from their pinned upstream sources and checks every reconstructed desktop file before activating the installation.

To build from a prepared AMD checkout on Linux, first use the ordinary build instructions to assemble `shellwork/win-x64`, stage the runtime in `build/runtime` and cache the 42 Comfy wheels in `build/.cache/comfy-wheels`. Download the pinned Lightricks Desktop source archive from the URL in `build_assets.py`; the builder verifies its full SHA-256. Then run the following from the repository root.

```sh
python3 setupwork/compact/build.py --source /path/to/desktop-upstream.zip
```

The generated installer is `build/release/LTX-AMD-395-MAX-Compact-Setup.exe`. The compact support files are compiled into the NSIS executable. A successful setup adds the normal Start menu shortcut and uninstaller. The existing installation stays available if a component fails verification before activation. The setup log is `%LOCALAPPDATA%\LTXDesktopAMD\setup\install.log`.
