# Setup and settings

## Platform requirements

Recommended specs, disk space and the hardware floors are in the README's [System requirements](../README.md#system-requirements). What each GPU and Mac size can run, the RAM each job needs, and Brave and Cautious modes are in [`hardware-compatibility.md`](hardware-compatibility.md).

## First run & data locations

LTX Desktop stores app data (settings, models, logs) in:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` (default: `~/.local/share/LTXDesktop/`)

Model weights are downloaded into the `models/` subfolder (this can be large and may take time).

On first launch, the Setup screen guides you through the steps below. It requires internet.

1. Sign in to Hugging Face and accept the license terms of [gated models](models-and-loras.md#gated-models) (the 2.5 weights are gated).
2. Optionally add an LTX API key for free cloud text encoding.
3. Choose the models to download and the models folder.
4. Optionally select **Opt out of basic analytics**.

The app shows a splash screen while it installs its Python environment.

Text encoding: to generate videos you must configure text encoding:

- **LTX API key** (cloud text encoding) — **text encoding via the API is completely FREE** and highly recommended to speed up inference and save memory. Generate a free API key at the [LTX Console](https://console.ltx.video/). [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).
- **Local Text Encoder** (extra download; enables fully-local operation on supported hardware) — if you don't wish to generate an API key, you can download the local encoder in **Settings > Models** and select it in **Settings > General > Text encoding**.

## Settings

Open **Settings** from the bottom of the sidebar.

| Tab | What it controls |
| --- | --- |
| **General** | Appearance (theme: system, light, or dark); auto-enhance for local and API generation; fast decode (faster video decode with slightly lower fidelity); text encoding (LTX API or local encoder); NVIDIA performance options (Torch Compile and diffusion stage cache, experimental); Apple Silicon performance options (faster Fast decode, experimental, Mac only); anonymous analytics |
| **Models** | Download and manage checkpoints, LoRAs, and encoder models. See [Managing models](models-and-loras.md#managing-models). Hidden in API-only mode |
| **Legacy** | Options for Gen Space and the Video Editor: project assets folder, default base model, preference for cloud video and image generation, and seed lock |
| **API Keys** | Optional LTX, fal, and Gemini keys. The app checks a key before it saves it, and you can remove a saved key |
| **About** | App version, the automatic update switch, and manual update checks |
