LTX Desktop - SpaceSpeak for Ubuntu (x86-64)
Original LTX Desktop by Lightricks. Community fork by Peter Beery / SpaceSpeak.com.

From the Downloads directory, run the ZIP directly as your regular desktop user:
  python3 LTX-Desktop-1.3.3-community.5-Linux-amd64-Online-Setup.zip

Use Ubuntu 22.04 or later on an x86-64 PC with an NVIDIA driver for local CUDA rendering. The bootstrap uses the system Python 3, dpkg-deb and apt. It downloads and verifies the pinned Lightricks Linux desktop and reconstructs this fork. Apt asks for permission to install it. Open LTX Desktop - SpaceSpeak from the applications menu. On first launch allow the Linux CUDA Python runtime download to finish.

To install MiniMax H3, open Settings > Models and choose FL2VA Base, FL2VA Turbo or Ref2VA. Review the linked MiniMax terms, confirm your authorization, and start setup. The application prepares its own ComfyUI engine and fetches the selected weights. Base and Turbo support Text to Video and Image to Video. Ref2VA takes reference images in Image to Video.

To prepare a verified .deb without installing it:
  python3 LTX-Desktop-1.3.3-community.5-Linux-amd64-Online-Setup.zip --output ./ltx-desktop-spacespeak.deb
