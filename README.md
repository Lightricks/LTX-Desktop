# LTX Desktop - AMD 395 MAX 1.3.2-amd.8

Windows x64 edition of LTX Desktop for Ryzen AI Max+ 395 PCs with Radeon 8060S graphics. The compact online installer sets up a private ComfyUI engine, the optional Kandinsky nodes, Python and the pinned AMD runtime. Recipients need a compatible AMD graphics driver and an internet connection to install the runtime and download their chosen model. LTX handles local engine setup from the app.

## Render from LTX Desktop - AMD 395 MAX

1. Extract the installer artifact on the target PC and run **LTX-AMD-395-MAX-Compact-Setup.exe**. For an upgrade, close the existing LTX app first. Setup recognizes the earlier **LTX Desktop AMD** installation and keeps your projects, settings, and downloaded models.
2. Open **Image to Video** or **Text to Video**, then open **Model**. Select the local model you want. Choosing a model that needs setup preserves the current prompt and image.
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

**Optional existing ComfyUI installation.** LTX can use a compatible AMD ComfyUI server already running on this PC on port 8000 or 8188, with the models in that server's folders. You can also choose **Use LTX-managed engine** in the setup panel. External Kandinsky installations use their own ComfyUI Manager and need core 0.38.0 or newer. The included engine already meets this requirement. Each model component retains its upstream license. [Kandinsky model and node guide](https://github.com/kandinskylab/kandinsky-6/blob/73b892b730885e02ab58fbf7fc5cb10b78e1e3da/comfyui/README.md#manual-downloads)

## Publish this repository

Extract the ZIP into a new folder. In GitHub Desktop, choose **File → Add local repository**, select the extracted `LTX-Desktop-AMD-395-MAX` folder, then choose **Publish repository**. In VS Code you can open the same folder and choose **Publish to GitHub**. The folder includes a clean, single-commit Git repository on `main`. Keep its hidden `.git` directory when extracting. You can publish the code before downloading any build dependencies.

This source snapshot contains the AMD app, all three MiniMax H3 integrations, license notices and compact online installer builder. Large upstream previews and two unmodified dependency archives are restored from URLs pinned in [source-assets.json](source-assets.json). The rest of the source is present and editable immediately. Each restored file is checked against its original SHA-256. The H3 weights are a separate, optional download handled by the application.

## Build the compact online installer

On GitHub, open **Actions → Build compact Windows AMD installer → Run workflow**. Download the workflow artifact when the run succeeds. It contains `LTX-AMD-395-MAX-Compact-Setup.exe` and its SHA-256 checksum.

For local development, restore the source assets once from the repository root using Python 3.11 or newer. On Windows use `py -3 restore_sources.py`. On Linux or WSL use `python3 restore_sources.py`. The first run requires internet access. Use `--verify-only` to check a restored checkout without downloading. The upstream downloads do not include any model weights.

To build on x86-64 Linux or WSL2, install Node.js 24 with npm and Corepack, Python 3.11+, curl, GNU coreutils, tar, xz, dpkg-deb and 7-Zip with CAB and NSIS support. Run these commands from the repository root:

```bash
python3 restore_sources.py
bash shellwork/build-shell.sh
python3 runtimework/prepare_runtime.py --mode web --out build/runtime
python3 setupwork/compact/build.py --source build/.cache/source-archives/ltx-desktop-upstream-d23b8b3de2ab258459f1c1524d4ce1e49c6783f5.zip
```

The unsigned Windows x64 installer is written under `build/release/`. End users run the installer on Windows. It retrieves and verifies the application components and Radeon runtime during setup. It provides its own ComfyUI engine.

## Source and verification

| Path | Contents |
|---|---|
| [app/](app/) | Desktop editor, Electron main process, FastAPI backend and upstream tests. Preview media is restored during build setup. |
| [vendor/LTX-2/](vendor/LTX-2/) | Complete LTX inference source, including the HIP attention changes, upstream license files, and training source. |
| [vendor/ComfyUI/](vendor/ComfyUI/) | Pinned upstream ComfyUI source and the optional Kandinsky node pack that ship with the application. |
| [shellwork/](shellwork/) | Locked JS build and Windows Electron shell assembly. |
| [runtimework/](runtimework/) | Python and ROCm dependency locks, acquisition and staging code, the three locally built wheels, and their rebuild instructions. |
| [setupwork/](setupwork/) | NSIS source, per-user installer engine, media builder and integrity checks. |
| [repair/](repair/) | Double-click offline repair for existing per-user AMD installations. |
| [checks/](checks/) | Installed-tree and GPU verification, with a receipt for the originally distributed shell. |

```bash
python3 -B -m unittest discover -s checks
python3 -B setupwork/test_media.py
python3 -B repair/test_repair.py
python3 -B runtimework/test_comfy_deps.py
```

To verify an installed build on Windows and exercise the model's attention operators on the Radeon, open PowerShell in the source repository and use the application's own interpreter.

```powershell
& "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\verify_install.py --gpu
```

To check the managed engine on a recipient's Radeon, run the following from the repository in PowerShell. It uses a temporary profile, performs a GPU calculation, installs the bundled Kandinsky nodes, and encodes and decodes a short video with sound. It needs no model weights.

```powershell
& "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_managed_comfy.py
```

The on-device [backend startup check](checks/check_backend_startup.py) launches the installed server in a temporary profile and verifies its authenticated health endpoint. A separate [Gemma check](checks/check_gemma_loading.py) loads and evaluates a tiny local checkpoint.

The earlier Radeon build passed a HIP BF16 matrix multiplication and attention probe on a physical 8060S. This release retains the Windows c10d startup repair. A timed render on that Windows machine is required to measure the new renderer’s actual speed.

## Upstream and notices

- Desktop source comes from Lightricks LTX Desktop v1.3.2, commit `d23b8b3de2ab258459f1c1524d4ce1e49c6783f5`. [Upstream Desktop](https://github.com/Lightricks/LTX-Desktop/tree/d23b8b3de2ab258459f1c1524d4ce1e49c6783f5)
- Inference source comes from Lightricks LTX-2 v1.6.0, commit `c1b4675863c07c59c92e502446cccdaedbe5889c`. [Upstream LTX-2](https://github.com/Lightricks/LTX-2/tree/c1b4675863c07c59c92e502446cccdaedbe5889c)
- The checkout includes upstream code licenses, model license texts, dependency notices and a list of the changes for this edition. [Project license](LICENSE) · [AMD changes](CHANGES-AMD.md) · [Desktop notices](app/NOTICES.md) · [LTX-2 license files](vendor/LTX-2/)
