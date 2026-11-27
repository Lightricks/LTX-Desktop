# Windows Radeon Python runtime

This directory builds the CPython 3.13.12 runtime used by LTX Desktop - AMD 395 MAX on Windows x64. The release targets the Radeon 8060S (gfx1151) with AMD's PyTorch 2.10.0, torchvision 0.25.0 and torchaudio 2.10.0 on ROCm 7.13.0. The AMD compatibility matrix lists Windows 11 25H2 and the Adrenalin 26.5.1 driver for this combination. The user must supply a working AMD graphics driver. [AMD compatibility](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html)

## Build inputs

Use CPython 3.11 or later and 7-Zip with CAB support on the packaging host. You can stage the Windows runtime on Linux or Windows. The build reads the approved URLs and SHA-256 digests from the two checked-in JSON files and verifies each file before use.

| Input | Origin |
|---|---|
| Windows embedded Python | The versioned python.org ZIP in `bootstrap-sources.json` |
| Four app-local Microsoft VC++ runtime DLLs | The immutable Microsoft VC++ 14.52.37029 x64 redistributable URL in `bootstrap-sources.json`. `visual_cpp.py` extracts the signed Cabinet payload and verifies each DLL's release digest |
| LTX dependencies | The Windows CPython 3.13 wheel selections carried forward from the LTX Desktop v1.3.2 backend lock (upstream v1.3.3 keeps the resolved package versions and moves the web server packages to its desktop extra), with the CUDA packages replaced by AMD's gfx1151 PyTorch, ROCm core and target libraries |
| LTX and ROCm metadata wheels | The three release wheels under `local-wheels`. Their source and rebuild commands are listed below |

Run one of the following commands from the repository root. Generated directories should be reserved for build output.

```text
python runtimework/prepare_runtime.py --out build/runtime-web --mode web
python runtimework/prepare_runtime.py --out build/runtime-offline --mode offline
```

Use `--sevenzip PATH` to select the 7-Zip executable. Use `--cache PATH` for a flat, reusable cache of downloaded files. A file already in that cache is used once its size and SHA-256 match the manifest. Running the web stage from an empty cache downloads approximately 30 MB. The offline stage additionally needs the public wheels, which total roughly 1.83 GB, and disk space for the expanded runtime. Changing the mode of a previously generated output directory replaces its previous contents after a successful build.

| Output | Web build | Offline build |
|---|---|---|
| `python/` | Windows interpreter and the four VC++ DLLs | Same |
| `vcredist/` | App-local runtime DLLs | Same |
| `wheels/` | Three locally built, hash-locked wheels | All 108 hash-locked wheels |
| `runtime-manifest.json`, `install_runtime.py`, `sitecustomize.py` | Installer inputs | Same |
| `python-embed/` | Created by the installer at installation time | Fully assembled runtime ready to archive |

The installer places the final Python environment in `application/resources/python`. Its startup hook registers the private ROCm DLL directories and the gfx1151 target. Model weights are downloaded by LTX after installation.

## Rebuild the release wheels

The checked-in wheel bytes let the installer build use exactly the tested release inputs. For independent verification, use uv 0.13.9 and run these commands from the repository root.

```text
python vendor/LTX-2/radeon/build_wheels.py --out-dir build/rebuilt-wheels
python runtimework/rebuild_rocm.py --out-dir build/rebuilt-wheels
```

The first command builds the patched `ltx-core` and matching `ltx-pipelines` from the vendored Lightricks v1.6.0 source. The second builds AMD's pure-Python `rocm` package from its original 7.13.0 source distribution with the gfx1151 target, setuptools 84.2.1 and a fixed build timestamp. Both commands compare the release outputs with their locked digests. When intentionally changing inference code, create new wheel versions and update the release lock and verification expectations together. [Lightricks source](https://github.com/Lightricks/LTX-2/tree/c1b4675863c07c59c92e502446cccdaedbe5889c) · [AMD source distribution](https://repo.amd.com/rocm/whl/gfx1151/rocm-7.13.0.tar.gz)

Upstream Lightricks licenses are included in `vendor/LTX-2`. The AMD source distribution preserves the Advanced Micro Devices copyright and MIT SPDX notices. The Windows embedded Python distribution includes its original license. Microsoft retains the terms of its redistributable. [Microsoft VC++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)


## Single-GPU startup

The Windows torch wheel for gfx1151 runs on the Radeon and reports that its native c10d distributed backend is unavailable. The installer applies a version- and hash-checked compatibility edit after extracting the pinned Transformers 5.14.1 and Diffusers 0.39.0 wheels. It guards distributed-only model loading and defers evaluation of device-mesh type annotations. The script verifies the installed source and updates its wheel RECORD entries. [Compatibility edits](repair_single_gpu.py)

To test the compatibility edits using the pinned release wheels, run `python3 runtimework/test_repair_single_gpu.py --wheelhouse /path/to/wheel-cache` from the repository root.

## Private ComfyUI engine packages

The shell build stages 42 Windows wheels for ComfyUI 0.38.0 and the Kandinsky 1.0.2 node pack. The locked downloads total 175,272,553 bytes. The installer includes the expanded files in `resources/comfy-engine/site-packages`, along with their distribution metadata and upstream license files. The app’s worker uses the existing Python 3.13.12 and ROCm 7.13 gfx1151 runtime. It can start without installing Python, Git or pip on the user's PC.

The engine loads its own PyAV 19.0.2 with FFmpeg 9 libraries, including H.264 and audio support. Native LTX continues using PyAV 16.1.0 in its own process. The official Windows `comfy-kitchen` 0.2.36 wheel includes an AMD HIP backend compiled for gfx1151 and an eager PyTorch implementation. The private engine includes the ComfyUI static frontend. LTX supplies the generation graphs, so the release omits ComfyUI's downloadable example-gallery media and offline documentation. [ComfyUI source for this release](https://github.com/Comfy-Org/ComfyUI/tree/6b747c0428c343e1417219641db93a4fb7cb69ae) · [Frontend source for 1.53.6](https://github.com/Comfy-Org/ComfyUI_frontend/tree/v1.53.6)

To stage only these wheels or verify the Windows dependency closure, run the commands below from the repository root. The verifier uses the build host's Python `packaging` module. Supplying `--base-wheels` also checks for conflicting FFmpeg DLLs in the native installation.

```shell
python runtimework/prepare_comfy_deps.py --out build/engine/site-packages --cache build/.cache/comfy-wheels
python runtimework/test_comfy_deps.py --wheelhouse build/.cache/comfy-wheels --base-wheels /path/to/native-wheel-cache
```
