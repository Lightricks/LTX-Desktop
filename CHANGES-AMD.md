# Changes for LTX Desktop - AMD 395 MAX

Original LTX Desktop by Lightricks. Community fork by Peter Beery / SpaceSpeak.com.

## 1.3.3-community.6 (macOS)

- The bundled Mac worker now attempts native Metal 4 INT8 multiplication for supported MiniMax H3 denoiser layers on the M5 family. It checks numerical correctness on the device before using the kernel and logs actual use during generation.
- Existing profiles reuse their H3 model files, shared encoders/decoders and license acknowledgement. Other eligible Apple GPUs run H3 through standard MPS. Windows uses CUDA or ROCm and Ubuntu uses CUDA. These installers remain on community.5.

## 1.3.3-community.5

- All desktop editions list FL2VA Base, FL2VA 8-step Turbo and Ref2VA. Apple silicon Metal and Ubuntu NVIDIA CUDA now offer the private engine, license prompt and model downloads. API-only hosts display the MiniMax catalog and hardware requirement.
- The Apple H3 workflow uses an FP16 video decoder and a Metal-compatible VAE path. CUDA and Radeon retain the existing INT8 recipes.
- The README lists model, runtime and hardware verification differences by deployment target.

## 1.3.3-community.4

- Apple silicon setup now preserves the signed frameworks and helpers from the official arm64 app. Setup reuses the previous verified download, signs the fork locally, verifies the assembled bundle and tests Metal on eligible Macs before installation.
- The Windows installer uses CPU-only PyTorch for the API profile selected in Parallels. Setup supplies the required Microsoft runtime libraries and exercises the API backend before activation. NVIDIA CUDA and Radeon ROCm keep their local renderers and shared model picker.
- Added a compact Ubuntu x86-64 online installer for native NVIDIA CUDA. First launch fetches the published Lightricks runtime pinned to the upstream version used by this fork.
- The launch screen identifies the community fork and platform GPU support. About credits Peter Beery / SpaceSpeak.com and names Lightricks as the original developer.

## 1.3.3-community.3

- Adds a native Apple silicon online installer for the community fork. The app uses Metal for local LTX on eligible M-series Macs, including M5 Pro. It uses its own macOS profile and bundle identity. Rerun the community installer to upgrade it.
- Windows setup retries missing-issuer certificate failures through Windows certificate verification and checks the expected size and SHA-256 before activating a downloaded component. Download errors now identify the cause in the installer details.
- The taskbar and Dock icons use the larger white atom, thicker orbit lines and larger gold lens flare.

## 1.3.3-community.2

The extended local model picker, model downloads and app-managed ComfyUI worker are available on supported Windows NVIDIA CUDA and Radeon ROCm installations. Both can select LTX 2.5 INT8, Kandinsky 6 Lite and MiniMax H3 alongside native LTX. H3 keeps its setup and usage consent checks. NVIDIA starts with native LTX selected and can switch renderers from the same model picker.

The Windows installer also recognizes the Radeon 8040S and 8050S alongside the 8060S and verifies GPU execution before enabling the Radeon runtime. The launch screen and Home identify the community fork with AMD GPU support. Windows users can open the fork credits through Help → About. The taskbar icon has a larger atom and thicker orbital lines.

## 1.3.3-community.1

- Merged the published Lightricks Desktop v1.3.3 release, including Keyframes, the shorter Extend/Retake context and Windows generation fixes. The LTX-2 inference dependency remains at upstream v1.6.0.
- Windows setup selects the upstream NVIDIA runtime or the Radeon gfx1151 runtime for the available hardware. Radeon installations retain the managed ComfyUI engine, model picker, downloads, license prompt, and ROCm compatibility changes.
- The application, Windows shortcuts and installer use a flat teal icon with a cyan atom and a gold lens-flare star in the upper right.
- The title screen identifies the community fork. Settings → About and the native About dialog identify the original project and credit Peter Beery / SpaceSpeak.com.
- Uses the existing Windows application identity, installation path and user data directories for upgrades from the AMD edition.

## 1.3.2-amd.8 local MiniMax H3 video

Image to Video adds MiniMax H3 FL2VA Base, FL2VA 8-step Turbo, and Ref2VA. Text to Video offers both FL2VA options. H3 renders audio and video together on the selected ComfyUI worker. Model setup in the generation screen and Settings shows the required files and reuses shared weights across variants. Local use requires the applicable MiniMax rights, which the operator confirms during setup.

## 1.3.2-amd.7 model setup readability

The model setup card in Image to Video and Text to Video, and the renderer controls in Settings → Models, now use compact, theme-aware text for model names, labels, and status messages.

## 1.3.2-amd.6 application rename

The application is now named **LTX Desktop - AMD 395 MAX** in its window, Windows shortcuts, installer and Installed apps entry. The installer upgrades the existing per-user installation. Existing projects, settings and downloaded models continue to use their original locations.

## 1.3.2-amd.5 self-contained local renderer

The online installer carries ComfyUI core 0.38.0, Kandinsky nodes 1.0.2, and the Windows packages needed to run their video and audio workflows with the included AMD Python runtime. The native and Comfy processes load their respective PyAV versions independently.

On a clean PC, the Model panel offers **Install engine & model**. LTX sets up the chosen model, checks availability and manages its own loopback server, including the restart needed to load Kandinsky. Its models and outputs live in the user's LTX data folder. Existing compatible ComfyUI installations are optional and can continue using their own model directories.

## 1.3.2-amd.4 in-app model setup

Image to Video and Text to Video open a setup panel for missing local models directly from the Model menu. Native LTX 2.3 and 2.5 offer a bundle download and activate the selected version after its required components are present. Independent native and Comfy downloads report their own progress.

The Comfy setup panel can download the LTX INT8 files or the seven Kandinsky Lite SD files into the running server’s model folders. Kandinsky checks its custom nodes separately from its weights and offers installation through the local ComfyUI Manager. Restart is a separate action. When setup succeeds, the picker refreshes the available models and selects the requested renderer.

## 1.3.2-amd.3 model selection and ComfyUI rendering

Image to Video and Text to Video can submit local jobs to AMD ComfyUI Desktop, retrieve the finished MP4, and preserve the renderer in the queue and history. The new default on the AMD edition is LTX 2.5 Fast with the Comfy INT8 ConvRot transformer and matching INT8 text encoder. The native engine and its workflows remain available in the picker. Kandinsky 6 Lite Distill uses its own image/text and audio workflow, checkpoint, and optional Comfy extension.

The LTX workflow is based on [Comfy’s LTX 2.5 image-to-video template](https://github.com/Comfy-Org/workflow_templates/blob/b0bcdb274878ea3c5cf46fed3819faa05b920707/templates/video_ltx2_5_i2v.json). The Kandinsky workflow uses the SD stage of the [Kandinsky Lab ComfyUI examples](https://github.com/kandinskylab/kandinsky-6/tree/73b892b730885e02ab58fbf7fc5cb10b78e1e3da/comfyui/example_workflows). Neither model weights nor the optional Comfy extension are redistributed with the installer.

## 1.3.2-amd.2 startup repair

On Windows, the previous build successfully ran BF16 matrix multiplication and attention on the Radeon 8060S, then stopped while loading the local model libraries. The pinned AMD PyTorch build omits the native c10d distributed backend. Model-library startup must check this capability before loading its optional multi-device features. The compatibility repair and runtime staging code for this release are in [runtimework/](runtimework/).

## 1.3.2-amd.1 desktop source changes

The Electron shell selects the private Python runtime and uses distinct AMD application data. The backend verifies HIP, BF16 matrix multiplication and scaled dot-product attention before it accepts local inference. The Hugging Face sign-in uses a loopback callback. The AMD edition disables the upstream binary updater.

Paths below are relative to `app/`.

- `electron/app-paths.ts`
- `electron/app-state.ts`
- `electron/gpu.ts`
- `electron/hf-loopback.test.ts`
- `electron/hf-loopback.ts`
- `electron/ipc/file-handlers.ts`
- `electron/main.ts`
- `electron/python-backend.ts`
- `electron/python-setup.ts`
- `electron/updater.ts`
- `frontend/components/settings/AboutSettingsSection.tsx`
- `frontend/contexts/AppSettingsContext.tsx`
- `frontend/hooks/use-hf-auth.ts`
- `backend/handlers/hf_auth_handler.py`
- `backend/ltx2_server.py`
- `backend/runtime_config/accelerator.py`
- `backend/services/gpu_info/gpu_info_impl.py`
- `backend/tests/test_accelerator.py`
- `backend/tests/test_hf_auth.py`

## Inference source changes

The core package adds a tiled PyTorch attention path on HIP and an eager neighborhood-attention implementation for the video decoder. The version of the custom wheel is `1.6.0+radeon1`. The pipelines wheel is built from the included upstream sources.

Paths below are relative to `vendor/LTX-2/`.

- `packages/ltx-core/pyproject.toml`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/video_vae/transformer/apply.py`
- `packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/__init__.py`
- `packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/eager.py`
- `packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/joint_eager.py`

## Packaging changes

The Windows shell assembler, ROCm dependency manifest, installer and diagnostics are provided alongside the application. The `radeon/` folder in the inference source contains wheel rebuild and numerical verification scripts. Three upstream symbolic links are stored as ordinary copies of their targets to make extraction on Windows reliable. These are `app/CLAUDE.md`, `app/frontend/remote/public/favicon.png`, and `vendor/LTX-2/packages/ltx-trainer/CLAUDE.md`.
