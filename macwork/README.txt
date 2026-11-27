LTX Desktop - SpaceSpeak
Apple Silicon online installation

Run this installer on the macOS side of your Mac. Local video generation uses Metal through PyTorch MPS. The application starts on macOS 14 or later. The native local renderer requires macOS 26 or later and at least 24 GiB unified memory. Older supported systems open in API mode.

1. Use Finder to expand the installer ZIP. Keep all the files together in the “LTX Desktop - SpaceSpeak Setup” folder.
2. Double-click “Install LTX Desktop.command”. A Terminal window shows the downloads and GPU check. If Finder declines to open the downloaded script, open Terminal, type /bin/bash followed by a space, drag “Install LTX Desktop.command” into the Terminal window, and press Return.
3. Wait for “Installed”. The installer opens LTX Desktop - SpaceSpeak from your home Applications folder. Download a native LTX model when prompted. The Windows managed ComfyUI extensions are available on the Windows edition.
4. To open the app again, choose Go > Home in Finder, then open Applications and “LTX Desktop - SpaceSpeak”. To upgrade, quit the existing SpaceSpeak app and run the new installer.

The online setup downloads the native LTX v1.3.3 Python environment and matching Electron 41.0.2 shell and verifies both release hashes. A Mac Developer ID notarization is not included. This installer assembles and signs its own app locally, then verifies the bundle. No system-wide security settings or administrator password are needed. The installation uses the current user's ~/Applications folder. Keeping the downloaded archives in ~/Library/Caches/com.spacespeak.ltx-desktop-community/setup/1.3.3 makes a retry quicker.

Peter Beery created this community branch of Lightricks’ LTX Desktop on behalf of SpaceSpeak.com.
