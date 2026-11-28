# LTX Desktop community fork

Original LTX Desktop by Lightricks. Community fork by Peter Beery / SpaceSpeak.com. The fork ships online installers for native Apple silicon, Windows x64 and Ubuntu x86-64. Windows supports NVIDIA CUDA and the gfx1151 Radeon 8040S, 8050S and 8060S in the Ryzen AI Max family. Existing AMD installations keep their app name, installation location and user data.

The Mac online installer is **1.3.3-community.6**. The Windows and Ubuntu online installers are **1.3.3-community.5**.

## Deployment differences in this release

All three desktop installers list **MiniMax H3 FL2VA Base**, **FL2VA 8-step Turbo**, and **Ref2VA** in **Settings → Models**. Base and Turbo work from **Text to Video** or **Image to Video**. Ref2VA needs one to nine reference images and appears in **Image to Video**. Installation uses the app’s private ComfyUI engine, even on a machine that has never had ComfyUI installed.

| Deployment target | Local native LTX 2.5 / 2.3 Fast | Local MiniMax H3 (all three variants) | LTX 2.5 Fast · ComfyUI INT8 | Kandinsky 6 Lite · ComfyUI |
| --- | --- | --- | --- | --- |
| Windows x64, supported NVIDIA CUDA | Yes | Yes, CUDA | Yes | Yes |
| Windows x64, Radeon 8040S / 8050S / 8060S (gfx1151) | Yes | Yes, ROCm | Yes | Yes |
| Native macOS, eligible M5 family including M5 Pro | Yes | Metal with native INT8 for supported H3 denoiser layers after an on-device check | Unavailable | Unavailable |
| Native macOS, other eligible Apple silicon | Yes | Standard PyTorch Metal (MPS) | Unavailable | Unavailable |
| Ubuntu 22.04+ x86-64, supported NVIDIA CUDA | Yes | Yes, CUDA | Unavailable | Unavailable |
| API-only installations, including the Parallels Display Adapter | Unavailable | Catalog only | Unavailable | Unavailable |

- **Mac.** Local tools require macOS 26+ and at least 24 GiB unified memory to pass the app’s eligibility check. This is the native LTX app threshold. Peak memory for a full H3 render on M5 Pro has not been measured. The Mac profile uses an FP16 video decoder and a Metal-compatible padding operation. The Mac build uses the bundled private worker for H3. Windows and Linux may also use an existing compatible local ComfyUI server when LTX can verify its GPU identity. The Mac installer is arm64. Intel Macs have no installer in this release.
- **Windows and Linux.** CUDA and Windows Radeon use the INT8 H3 video decoder. Linux ships an amd64 Ubuntu installer for NVIDIA. This release has no Linux Radeon, Linux arm64 or native Windows-on-Arm local-GPU installer. The Windows x64 app in Parallels can use the cloud interface.
- **Cloud and local workflows.** Gen Space uses the configured LTX service on all three desktop editions. LTX Pro is a cloud model. MiniMax H3 uses the local worker and has no cloud fallback in this fork. Each qualifying user must confirm the MiniMax license before setting up H3. Downloads are optional. Windows provides the managed Kandinsky node installer and GPU diagnostics shortcut. A paired remote client uses its host’s installed models. Manage downloads and license consent on the desktop host. **Extend**, **Retake**, **Audio to Video** and custom LoRA workflows use their native LTX model options. H3 is offered for text- and image-guided video.
- **Hardware limits.** Duration, resolution and advanced native LTX tools depend on GPU memory, system memory and storage. Availability in the table describes the implemented software paths. All three H3 graphs passed the pinned ComfyUI validator. A weight-free four-frame H.264 clip with AAC audio exercised the live Linux worker and output download on CPU. End-to-end generation with actual H3 weights on M5 Pro, Linux CUDA and either Windows GPU remains unverified for this release. [Upstream hardware guide](app/docs/hardware-compatibility.md)

Version 1.3.3 includes upstream Keyframes, a shorter context option for faster Extend and Retake, and Windows fixes for Gemma loading and Smart App Control. [Lightricks 1.3.3 release notes](https://github.com/Lightricks/LTX-Desktop/releases/tag/v1.3.3)

## Install on an Apple silicon Mac

On an M5 Pro, run the Mac installer from Finder in macOS. The Windows installer recognizes the Parallels Display Adapter as an API-only device. Native macOS uses PyTorch Metal (MPS) and the Mac’s unified memory.

1. Check **Apple menu → About This Mac** for the chip, memory and macOS version. Local generation needs macOS 26 or later and at least 24 GiB of unified memory. Lightricks recommends 32 GB or more. Allow at least 100 GB of free storage for local models. Macs with macOS 14 or 15, or less than 24 GiB of memory, can use the app in API mode. [Upstream Mac requirements](https://github.com/Lightricks/LTX-Desktop#system-requirements)
2. Quit the SpaceSpeak app if it is running. Download **LTX-Desktop-1.3.3-community.6-Apple-Silicon-Online-Setup.zip** and double-click the ZIP to extract it. Open **LTX Desktop - SpaceSpeak Setup** and double-click **Install LTX Desktop.command**. Keep the extracted folder together during setup. Setup verifies and reuses the official 836 MB Mac archive from an earlier attempt when available. It preserves the signed Electron frameworks and helpers, checks the finished bundle and tests Metal on an eligible Mac before activating the installation.
3. Open **LTX Desktop - SpaceSpeak** in your home **Applications** folder. Open **Settings → Models** to download a native LTX model. The Mac build supports native LTX 2.5 Fast and 2.3 Fast. For H3, select a MiniMax entry in **Settings → Models**, review the license, then use the setup button to prepare the worker and download the weights. See the deployment table above for the Windows-only renderers.
4. For a first local test, pick **LTX 2.5 Fast**, a five-second clip and a low resolution. The app stores its community profile under `~/Library/Application Support/LTXDesktopSpaceSpeak`.

The Mac installer assembles the community app from pinned components and signs it locally. This release has no Apple Developer ID notarization. If macOS requests approval for an identified item you trust, use the per-item control under **System Settings → Privacy & Security**. [Apple’s guide to opening apps](https://support.apple.com/en-us/102445)

### MiniMax H3 on the M5 family

The Mac worker tests the bundled Metal 4 kernel on the device. Once it passes, eligible INT8 ConvRot linear layers in the H3 denoiser use the M5 GPU’s integer matrix multiplication for FL2VA Base, FL2VA 8-step Turbo and Ref2VA. The worker uses standard MPS for unsupported operations and if the on-device check fails. Prompt encoding, audio/video decoding and native LTX have their own execution paths. Installing this upgrade reuses the existing model downloads and license acknowledgement. The bundled PyTorch runtime compiles the shader on the Mac. The installer includes everything this kernel needs.

To check a render, run the same prompt with a new seed to avoid an output-cache hit. Open `~/Library/Application Support/LTXDesktopSpaceSpeak/managed-comfy/worker.log` before restarting the app. Search for `[SpaceSpeak H3 Metal]`. **Native INT8 active** records a real denoiser layer using the kernel. **Self-test passed** confirms only the synthetic check. **Standard MPS** explains why the worker used its normal path. For a timing comparison, record the resolution, duration, sampling time and total time. W8A8 also quantizes activations, so small output differences are possible even with a fixed seed. A full-weight M5 Pro benchmark for this fork is pending.

To compare against standard MPS, quit the app and launch it once from Terminal with the optimization disabled. Normal Finder launches use automatic selection.

```bash
LTX_H3_M5_INT8=0 "$HOME/Applications/LTX Desktop - SpaceSpeak.app/Contents/MacOS/LTX Desktop"
```

The extracted installer also includes `check_h3_m5_kernel.py`. Once the app’s private engine is set up, open Terminal in the installer folder and run the following command. It checks a synthetic projection and compares the median time for a representative H3 layer using stock MPS and native INT8. The timings cover that layer only.

```bash
"$HOME/Applications/LTX Desktop - SpaceSpeak.app/Contents/Resources/python/bin/ltx-comfy-python" -I -B check_h3_m5_kernel.py --benchmark
```

## Install on Ubuntu with NVIDIA CUDA

On an x86-64 PC running Ubuntu 22.04 or later, install the NVIDIA driver and download the Linux online installer. From your Downloads folder, run it as your regular desktop user. The system asks for permission when it installs the Debian package.

```bash
python3 LTX-Desktop-1.3.3-community.5-Linux-amd64-Online-Setup.zip
```

Open **LTX Desktop - SpaceSpeak** from the applications menu. First launch downloads the pinned Linux runtime. Under **Settings → Models**, choose a native LTX model. MiniMax H3 setup is also available in this panel. The Ubuntu package includes the private ComfyUI core and downloads the matching Linux worker dependencies on first setup. [Upstream Linux requirements](https://github.com/Lightricks/LTX-Desktop#system-requirements)

## How Windows setup selects a GPU

| Hardware detected at setup | Rendering path |
| --- | --- |
| Supported NVIDIA GPU | Native LTX on CUDA by default, plus the managed ComfyUI renderer, model picker, and downloads. On systems with both supported targets, setup selects NVIDIA. |
| Radeon 8040S, 8050S or 8060S without a qualifying NVIDIA GPU | The existing ROCm runtime with native LTX, the managed ComfyUI engine, in-app model setup, and optional Kandinsky and MiniMax H3 workflows. |
| Neither local target is available | API-only mode using a CPU runtime. This includes the Parallels Display Adapter. Remote generation requires the appropriate service credentials. |

The bundled Windows ROCm and PyTorch libraries target gfx1151. Setup checks the selected Radeon with a short GPU calculation before activating the installation. AMD lists the 8040S, 8050S and 8060S under this architecture. [AMD ROCm 7.13 hardware list](https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html)

Install the GPU vendor's driver before running setup. The installer selects the runtime automatically. Rerun setup after switching graphics hardware. The NVIDIA runtime download is approximately 3.30 GB before model weights. API mode uses roughly 365 MB of CPU runtime wheels and checks that the API backend initializes. In the Windows Start menu, open **LTX Desktop - AMD 395 MAX Diagnostics** to see the selected GPU and run a compute check. Its log is saved at `%LOCALAPPDATA%\LTXDesktopAMD\logs\gpu-diagnostic.txt`. On either local GPU path, the app can set up its own ComfyUI engine on a fresh PC. Model weights download when selected.

The launch screen and Home identify this as a community fork and name the platform’s GPU backend. On Windows, choose **Help → About** for the app version, upstream project, and fork credits.

## Render from LTX Desktop - AMD 395 MAX

1. Run the Windows online installer. For an upgrade, close the existing LTX app first. If you built it using GitHub Actions, extract **LTX-AMD-395-MAX-Compact-Setup.exe** from the workflow artifact. Setup recognizes the earlier **LTX Desktop AMD** installation and keeps your projects, settings, and downloaded models.
2. On NVIDIA or Radeon, open **Image to Video** or **Text to Video**, then open **Model**. Select the local model you want. Choosing a model that needs setup preserves the current prompt and image.
3. For **LTX 2.5 Fast · ComfyUI INT8** or **Kandinsky 6 Lite Distill · ComfyUI** on a new computer, click **Install engine & model**. LTX prepares the bundled engine, starts it locally and downloads the model. For Kandinsky it also installs the bundled node pack and restarts its own engine once idle. The panel shows progress. If LTX asks for access to a gated model, follow the Hugging Face sign-in and terms links, then click **Retry model setup**.
4. For **LTX 2.3 Fast · Native** or **LTX 2.5 Fast · Native**, click **Download & select**. To switch to an installed inactive native model, choose **Activate**, then **Use this model**.
5. Select **Use this model** if needed, then **Generate**. The first generation loads the selected weights. A five-second clip at 24 fps is a convenient first test.

| Model option | Intended use |
|---|---|
| LTX 2.5 Fast · ComfyUI INT8 | Local joint image/text-to-video and audio. Prompt enhancement is off. At 270p and 360p it uses one sampling stage. Higher resolutions add latent upscaling and a refinement pass. |
| LTX 2.5 Fast or 2.3 Fast · Native | Use the native LTX libraries and the matching downloaded checkpoint. Select Native for end frames, custom LoRAs and negative-attention guidance. |
| Kandinsky 6 Lite Distill · ComfyUI | Optional 3B model for 5 seconds at 24 fps, landscape 864×480 or portrait 480×864, with synchronized audio. Its prompt uses the same description for picture and sound. |
| MiniMax H3 · FL2VA Base | Generate a clip with sound from a description, optionally guided by a first frame, last frame, or both. Uses 20 sampling steps. |
| MiniMax H3 · FL2VA 8-step Turbo | Use the same inputs and base weights with an additional adapter that reduces sampling to eight steps. |
| MiniMax H3 · Ref2VA | In Image to Video, guide the appearance of subjects or scenes with one to nine ordered reference images. This is a separate 20-step model. |
| Cloud Fast and Pro models in Gen Space | Opens the existing project and API generation interface. Paid generation requires your configured LTX account. |

A fresh LTX INT8 model download is 39.71 GB. A fresh Kandinsky Lite model download is 25.42 GB. On Windows, LTX keeps the managed engine's weights in `%LOCALAPPDATA%\LTXDesktopAMD\managed-comfy\base\models`. These files are separate from the program installation and survive upgrades. The engine's startup log is `%LOCALAPPDATA%\LTXDesktopAMD\managed-comfy\worker.log`.

**Using MiniMax H3.** Select an H3 model in the generation screen or under **Settings → Models**. Read the license and confirm that your use is covered before starting the download. MiniMax's community license excludes the United States, the United Kingdom, the European Union and South Korea. Users in these territories need separate authorization from MiniMax for local use. The installed app includes the integration and the license. Model weights download only when requested. [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

A fresh FL2VA Base or Ref2VA download needs **51.53 GB** on Windows or Linux and **53.92 GB** on macOS. These are download sizes. Peak memory use during rendering varies. After FL2VA Base, the Turbo adapter adds **1.96 GB** and Ref2VA adds **20.97 GB**. The three variants share the text encoder and both audio and video decoders. All three support 24 fps and offer 5-, 10-, or 15-second targets, at 480p or 768p. Start with the 5-second, 480p setting to evaluate render time on your hardware. Describe the picture and audio together in the prompt.

**Optional existing ComfyUI installation on Windows or Linux.** LTX can use a compatible ComfyUI server on the selected GPU already running on this PC on port 8000 or 8188, with the models in that server's folders. If LTX cannot identify that server’s GPU, choose **Use LTX-managed engine** to run on the adapter verified by setup. External Kandinsky installations use their own ComfyUI Manager and need core 0.38.0 or newer. The included engine already meets this requirement. Each model component retains its upstream license. [Kandinsky model and node guide](https://github.com/kandinskylab/kandinsky-6/blob/73b892b730885e02ab58fbf7fc5cb10b78e1e3da/comfyui/README.md#manual-downloads)

## Publish this code to your GitHub fork

1. Sign in to GitHub and select **Fork → Create fork** on [Lightricks/LTX-Desktop](https://github.com/Lightricks/LTX-Desktop). If you already have a fork, open it. Copy its HTTPS URL from **Code**.
2. For the original `v1.3.3-community.5` compact source distribution, extract the compact source ZIP into a new folder, including its hidden `.git` directory. Open a terminal inside the extracted repository. Confirm that `git status --short` is empty. The release tag names the exact tree used to build the installers.
3. Run the following once in this extracted checkout. Replace `YOUR-ACCOUNT` with your GitHub username or organization. These commands connect the compact source history to its exact Lightricks base and place the work on a new branch in your fork.

```bash
git switch -c spacespeak-gpu v1.3.3-community.5
git remote add origin https://github.com/YOUR-ACCOUNT/LTX-Desktop.git
git remote add upstream https://github.com/Lightricks/LTX-Desktop.git
git fetch --no-tags upstream refs/tags/v1.3.3
git merge --allow-unrelated-histories -s ours -m "Connect SpaceSpeak fork to Lightricks v1.3.3" 2ab055b98f7590581c56f9e0b2ff55ac56d459b7
git diff --exit-code v1.3.3-community.5 HEAD
```

4. An empty diff confirms that the history connection preserved the release files. Run `git push -u origin spacespeak-gpu` to publish your revision. On GitHub, open **Branches → spacespeak-gpu** in your fork. You can make this your fork's default branch in its settings.

The repository includes the fork source, build scripts, platform installers' source, upstream notices and all three MiniMax integrations for eligible Windows, Linux and macOS GPUs. Build preparation restores large upstream media and two dependency archives from pinned URLs and verifies their SHA-256 digests. Model weights download when requested in the application. [Pinned source assets](source-assets.json)

## Build the native Mac online installer

To build and test on GitHub, open **Actions → Build and check Apple silicon online installer** and run the workflow. It builds the app, assembles the compact installer, installs on a macOS arm64 runner and checks application startup. The runner verifies Metal execution when its OS, memory and GPU support local generation. The same on-device checks run in the installer on an eligible physical Mac. The Mac builder and end-user instructions are in [macwork](macwork/).

## Build the Linux online installer

On x86-64 Linux with the source assets restored, Node 24, pnpm from Corepack and the Debian packaging tools installed, build the application and its Debian package.

```bash
(cd app && corepack pnpm install --frozen-lockfile && corepack pnpm run build:frontend && corepack pnpm exec electron-builder --linux deb --x64 --config electron-builder.linux.cjs --publish never)
curl -fL --retry 3 -o LTX-Desktop-amd64.deb https://github.com/Lightricks/LTX-Desktop/releases/download/v1.3.3/LTX-Desktop-amd64.deb
python3 linuxwork/build_online.py --upstream-deb LTX-Desktop-amd64.deb --deb app/release/LTX-Desktop-SpaceSpeak-1.3.3-community.5-amd64.deb --output LTX-Desktop-1.3.3-community.5-Linux-amd64-Online-Setup.zip
python3 -B linuxwork/test_online.py
```

The builder checks the pinned upstream digest. The end-user installer verifies the reconstructed Debian control and application archives before installing.

## Build the Windows online installer

On GitHub, open **Actions → Build compact Windows GPU installer → Run workflow**. Download the workflow artifact when the run succeeds. It contains `LTX-AMD-395-MAX-Compact-Setup.exe` and its SHA-256 checksum.

For local development, restore the source assets once from the repository root using Python 3.11 or newer. On Windows use `py -3 restore_sources.py`. On Linux or WSL use `python3 restore_sources.py`. The first run requires internet access. Use `--verify-only` to check a restored checkout without downloading. The upstream downloads do not include any model weights.

To build on x86-64 Linux or WSL2, install Node.js 24 with npm and Corepack, Python 3.11+, curl, GNU coreutils, tar, xz, dpkg-deb and 7-Zip with CAB and NSIS support. Run these commands from the repository root:

```bash
python3 restore_sources.py
bash shellwork/build-shell.sh
python3 runtimework/prepare_runtime.py --mode web --out build/runtime
python3 setupwork/compact/build.py --source build/.cache/source-archives/ltx-desktop-upstream-2ab055b98f7590581c56f9e0b2ff55ac56d459b7.zip
```

The unsigned Windows x64 installer is written under `build/release/`. End users run the installer on Windows. Setup retrieves and verifies the application components and the runtime for the selected GPU.

## Source and verification

| Path | Contents |
|---|---|
| [app/](app/) | Desktop editor, Electron main process, FastAPI backend and upstream tests. Preview media is restored during build setup. |
| [vendor/LTX-2/](vendor/LTX-2/) | Complete LTX inference source, including the HIP attention changes, upstream license files, and training source. |
| [vendor/ComfyUI/](vendor/ComfyUI/) | Pinned upstream ComfyUI source and the optional Kandinsky node pack that ship with the application. |
| [shellwork/](shellwork/) | Locked JS build and Windows Electron shell assembly. |
| [runtimework/](runtimework/) | Python and ROCm dependency locks, acquisition and staging code, the three locally built wheels, and their rebuild instructions. |
| [setupwork/](setupwork/) | Windows NSIS source, per-user installer engine and integrity checks. |
| [macwork/](macwork/) | Native Apple silicon online installer, app assembly and upgrade tests. |
| [linuxwork/](linuxwork/) | Ubuntu x86-64 online installer, compact package builder and reconstruction tests. |
| [repair/](repair/) | Double-click offline repair for existing per-user AMD installations. |
| [checks/](checks/) | Installed-tree and GPU verification, with a receipt for the originally distributed shell. |

```bash
python3 -B -m unittest discover -s checks
python3 -B setupwork/test_gpu_installer.py
python3 -B setupwork/test_media.py
python3 -B repair/test_repair.py
python3 -B runtimework/test_comfy_deps.py
```

To verify an installed build on Windows and exercise the selected GPU, open PowerShell in the source repository and use the application's own interpreter.

```powershell
& "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\verify_install.py --gpu
```

To check the managed engine on a Windows NVIDIA or Radeon GPU, run the following from the repository in PowerShell. It uses a temporary profile, performs a GPU calculation, installs the bundled Kandinsky nodes, and encodes and decodes a short video with sound. It needs no model weights.

```powershell
& "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_managed_comfy.py
```

On macOS or Linux the same script accepts `--resources` and uses the installed desktop Python. It checks H3 node registration and exercises video/audio output without fetching model weights. On macOS the following command uses the app in your home Applications folder.

```bash
resources="$HOME/Applications/LTX Desktop - SpaceSpeak.app/Contents/Resources"
"$resources/python/bin/python3" -B checks/check_managed_comfy.py --resources "$resources"
```

The on-device [backend startup check](checks/check_backend_startup.py) launches the installed server in a temporary profile and verifies its authenticated health endpoint. A separate [Gemma check](checks/check_gemma_loading.py) loads and evaluates a tiny local checkpoint.

The earlier Radeon build passed a HIP BF16 matrix multiplication and attention probe on a physical 8060S. This release retains the Windows c10d startup repair. A timed render on that Windows machine is required to measure the new renderer’s actual speed.

## Upstream and notices

- Desktop source comes from Lightricks LTX Desktop v1.3.3, commit `2ab055b98f7590581c56f9e0b2ff55ac56d459b7`. [Upstream Desktop](https://github.com/Lightricks/LTX-Desktop/tree/2ab055b98f7590581c56f9e0b2ff55ac56d459b7)
- Inference source comes from Lightricks LTX-2 v1.6.0, commit `c1b4675863c07c59c92e502446cccdaedbe5889c`. [Upstream LTX-2](https://github.com/Lightricks/LTX-2/tree/c1b4675863c07c59c92e502446cccdaedbe5889c)
- The checkout includes upstream code licenses, model license texts, dependency notices and a list of the changes for this edition. [Project license](LICENSE) · [AMD changes](CHANGES-AMD.md) · [Desktop notices](app/NOTICES.md) · [LTX-2 license files](vendor/LTX-2/)
