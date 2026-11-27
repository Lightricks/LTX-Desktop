# LTX Desktop community edition 1.3.3-community.3

Peter Beery created this community branch of Lightricks’ LTX Desktop on behalf of SpaceSpeak.com. The Windows x64 installer supports the original NVIDIA CUDA renderer and Radeon 8040S, 8050S and 8060S graphics in the Ryzen AI Max family. These three GPUs use the gfx1151 architecture. Existing AMD installations keep their app name, installation location and user data.

On supported Windows GPUs, open **Settings → Models** to manage the native LTX models, the ComfyUI INT8 renderer, Kandinsky Lite, and MiniMax H3. You can also select and set up a model directly in **Image to Video** or **Text to Video**. H3 Ref2VA takes reference images and appears under **Image to Video**.

Version 1.3.3 includes upstream Keyframes, a shorter context option for faster Extend and Retake, and Windows fixes for Gemma loading and Smart App Control. [Lightricks 1.3.3 release notes](https://github.com/Lightricks/LTX-Desktop/releases/tag/v1.3.3)

## Install on an Apple silicon Mac

On an M5 Pro, run the Mac installer from Finder in macOS. The Windows installer recognizes the Parallels Display Adapter as an API-only device. Native macOS uses PyTorch Metal (MPS) and the Mac’s unified memory.

1. Check **Apple menu → About This Mac** for the chip, memory and macOS version. Local generation needs macOS 26 or later and at least 24 GiB of unified memory. Lightricks recommends 32 GB or more. Allow at least 100 GB of free storage for local models. Macs with macOS 14 or 15, or less than 24 GiB of memory, can use the app in API mode. [Upstream Mac requirements](https://github.com/Lightricks/LTX-Desktop#system-requirements)
2. Download **LTX-Desktop-1.3.3-community.3-Apple-Silicon-Online-Setup.zip** and double-click the ZIP to extract it. Open **LTX Desktop - SpaceSpeak Setup** and double-click **Install LTX Desktop.command**. Keep the extracted folder together during setup. The installer downloads and verifies the native runtime components. On a Mac eligible for local generation it checks Metal execution before installing.
3. Open **LTX Desktop - SpaceSpeak** in your home **Applications** folder. Open **Settings → Models** to download a native LTX model. The Mac build supports native LTX 2.5 Fast and 2.3 Fast. The managed ComfyUI, Kandinsky and MiniMax H3 integrations in this release run on supported Windows GPUs.
4. For a first local test, pick **LTX 2.5 Fast**, a five-second clip and a low resolution. The app stores its community profile under `~/Library/Application Support/LTXDesktopSpaceSpeak`.

The Mac installer assembles the community app from pinned components and signs it locally. This release has no Apple Developer ID notarization. If macOS requests approval for an identified item you trust, use the per-item control under **System Settings → Privacy & Security**. [Apple’s guide to opening apps](https://support.apple.com/en-us/102445)

## How Windows setup selects a GPU

| Hardware detected at setup | Rendering path |
| --- | --- |
| Supported NVIDIA GPU | Native LTX on CUDA by default, plus the managed ComfyUI renderer, model picker, and downloads. On systems with both supported targets, setup selects NVIDIA. |
| Radeon 8040S, 8050S or 8060S without a qualifying NVIDIA GPU | The existing ROCm runtime with native LTX, the managed ComfyUI engine, in-app model setup, and optional Kandinsky and MiniMax H3 workflows. |
| Neither local target is available | The upstream API-only mode. Remote generation requires the appropriate service credentials. |

The bundled Windows ROCm and PyTorch libraries target gfx1151. Setup checks the selected Radeon with a short GPU calculation before activating the installation. AMD lists the 8040S, 8050S and 8060S under this architecture. [AMD ROCm 7.13 hardware list](https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html)

Install the GPU vendor's driver before running setup. The installer selects the runtime automatically. Rerun setup after switching graphics hardware. The NVIDIA runtime download is approximately 3.30 GB before model weights. In the Windows Start menu, open **LTX Desktop - AMD 395 MAX Diagnostics** to see the selected GPU and run a compute check. Its log is saved at `%LOCALAPPDATA%\LTXDesktopAMD\logs\gpu-diagnostic.txt`. On either local GPU path, the app can set up its own ComfyUI engine on a fresh PC. Model weights download when selected.

The launch screen and Home identify this as the AMD GPU support branch. On Windows, choose **Help → About** for the app version, upstream project, and branch credits.

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

A fresh LTX INT8 model download is 39.71 GB. A fresh Kandinsky Lite model download is 25.42 GB. LTX keeps the managed engine's weights in `%LOCALAPPDATA%\LTXDesktopAMD\managed-comfy\base\models`. These files are separate from the program installation and survive upgrades. The engine's startup log is `%LOCALAPPDATA%\LTXDesktopAMD\managed-comfy\worker.log`.

**Using MiniMax H3.** Select an H3 model in the generation screen or under **Settings → Models**. Read the license and confirm that your use is covered before starting the download. MiniMax's community license excludes the United States, the United Kingdom, the European Union and South Korea. Users in these territories need separate authorization from MiniMax for local use. The installed app includes the integration and the license. Model weights download only when requested. [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

FL2VA Base or Ref2VA needs **51.53 GB** from a fresh installation. After FL2VA Base, the Turbo adapter adds **1.96 GB** and Ref2VA adds **20.97 GB**. The three variants share the text encoder and both audio and video decoders. All three support 24 fps and offer 5-, 10-, or 15-second targets, at 480p or 768p. Start with the 5-second, 480p setting to evaluate render time on your hardware. Describe the picture and audio together in the prompt.

**Optional existing ComfyUI installation.** LTX can use a compatible ComfyUI server on the selected GPU already running on this PC on port 8000 or 8188, with the models in that server's folders. If LTX cannot identify that server’s GPU, choose **Use LTX-managed engine** to run on the adapter verified by setup. External Kandinsky installations use their own ComfyUI Manager and need core 0.38.0 or newer. The included engine already meets this requirement. Each model component retains its upstream license. [Kandinsky model and node guide](https://github.com/kandinskylab/kandinsky-6/blob/73b892b730885e02ab58fbf7fc5cb10b78e1e3da/comfyui/README.md#manual-downloads)

## Publish this repository

Extract the ZIP into a new folder. In GitHub Desktop, choose **File → Add local repository**, select the extracted `LTX-Desktop-AMD-395-MAX` folder, then choose **Publish repository**. In VS Code you can open the same folder and choose **Publish to GitHub**. The folder includes a Git repository on `main`. Keep its hidden `.git` directory when extracting. You can publish the code before downloading any build dependencies.

This source snapshot contains the Windows and native Apple silicon app, all three MiniMax H3 integrations, license notices and compact online installer builder. Large upstream previews and two unmodified dependency archives are restored from URLs pinned in [source-assets.json](source-assets.json). The rest of the source is present and editable immediately. Each restored file is checked against its original SHA-256. The H3 weights are a separate, optional download handled by the application.

## Build the native Mac online installer

To build and test on GitHub, open **Actions → Build and check Apple silicon online installer** and run the workflow. It builds the app, assembles the compact installer, installs on a macOS arm64 runner and checks application startup. The runner verifies Metal execution when its OS, memory and GPU support local generation. The same on-device checks run in the installer on an eligible physical Mac. The Mac builder and end-user instructions are in [macwork](macwork/).

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

To check the managed engine on a recipient’s NVIDIA or Radeon GPU, run the following from the repository in PowerShell. It uses a temporary profile, performs a GPU calculation, installs the bundled Kandinsky nodes, and encodes and decodes a short video with sound. It needs no model weights.

```powershell
& "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_managed_comfy.py
```

The on-device [backend startup check](checks/check_backend_startup.py) launches the installed server in a temporary profile and verifies its authenticated health endpoint. A separate [Gemma check](checks/check_gemma_loading.py) loads and evaluates a tiny local checkpoint.

The earlier Radeon build passed a HIP BF16 matrix multiplication and attention probe on a physical 8060S. This release retains the Windows c10d startup repair. A timed render on that Windows machine is required to measure the new renderer’s actual speed.

## Upstream and notices

- Desktop source comes from Lightricks LTX Desktop v1.3.3, commit `2ab055b98f7590581c56f9e0b2ff55ac56d459b7`. [Upstream Desktop](https://github.com/Lightricks/LTX-Desktop/tree/2ab055b98f7590581c56f9e0b2ff55ac56d459b7)
- Inference source comes from Lightricks LTX-2 v1.6.0, commit `c1b4675863c07c59c92e502446cccdaedbe5889c`. [Upstream LTX-2](https://github.com/Lightricks/LTX-2/tree/c1b4675863c07c59c92e502446cccdaedbe5889c)
- The checkout includes upstream code licenses, model license texts, dependency notices and a list of the changes for this edition. [Project license](LICENSE) · [AMD changes](CHANGES-AMD.md) · [Desktop notices](app/NOTICES.md) · [LTX-2 license files](vendor/LTX-2/)
