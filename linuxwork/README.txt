LTX Desktop - SpaceSpeak for Ubuntu (x86-64)
Original LTX Desktop by Lightricks. Community fork by Peter Beery / SpaceSpeak.com.

From the Downloads directory, run the ZIP directly as your regular desktop user:
  python3 LTX-Desktop-1.3.3-community.4-Linux-amd64-Online-Setup.zip

Use Ubuntu 22.04 or later on an x86-64 PC with an NVIDIA driver for local CUDA rendering. The bootstrap uses the system Python 3, dpkg-deb and apt. It downloads and verifies the pinned Lightricks Linux desktop and reconstructs this fork. Apt asks for permission to install it. Open LTX Desktop - SpaceSpeak from the applications menu. On first launch allow the Linux CUDA Python runtime download to finish. Use Settings > Models to download native LTX models. Model weights are separate. The managed Windows ComfyUI model choices run on Windows.

To prepare a verified .deb without installing it:
  python3 LTX-Desktop-1.3.3-community.4-Linux-amd64-Online-Setup.zip --output ./ltx-desktop-spacespeak.deb
