# LTX-2 1.5.0 upgrade: analysis and decisions

This document records the move from `ltx-core` and `ltx-pipelines` v1.4.1 to v1.5.0
(release notes: <https://github.com/Lightricks/LTX-2/releases/tag/v1.5.0>). Version 1.5.0 includes 1.4.2.
The manual test plan is in `docs/ltx-2-1-5-0-manual-test-plan.md`.

## What changed in Desktop

| Area | Change |
| --- | --- |
| Pins | `backend/pyproject.toml` pins `ltx-core==1.5.0`, `ltx-pipelines==1.5.0`, tag `v1.5.0`, and the two `dependency-metadata` versions. |
| Lock | `backend/uv.lock` has the new version and commit `e98f6d927d83c92ef2be6aec76c8440b19306e39`. No dependency changed. |
| Image inputs | One helper, `to_ltx_images` in `services/ltx_pipeline_common.py`, converts Desktop stills to upstream inputs. Fast, A2V and IC-LoRA use it. |
| Fast pipeline | `guide_all_images` now selects `ImageConditioningInput.guide` for every still. The helper swap in `distilled_keyframe_guiding.py` is deleted. |
| Skip stage 2 | `_call_stage_1_only` calls `load_conditionings` and `image_conditionings`. |
| A2V | `_prepare` calls `load_conditionings`. Both stages and every window encode the loaded stills. `resolve_image_conditionings` and `encode_stage_image_conditionings` are deleted. |
| Tiled IC-LoRA | Stills at frame -1 use `image_conditionings`. The chunk call has no `color_space`. |
| Patches | All kept. Version comments updated. The `diffusion_stage_cache` test fake follows the new `DiffusionStage` surface. |

## Upstream API map

| 1.4.1 | 1.5.0 |
| --- | --- |
| `ImageConditioningInput(path, frame_idx, strength)` | `ImageConditioningInput.at / .replace / .guide(path, frame_idx, strength, crf=None)` |
| `ImageConditioner.resolve_crf` | `ImageConditioner.load_conditionings(images, color_space)` |
| `combined_image_conditionings(...)` | `image_conditionings(loaded_images, height, width, video_encoder, dtype, device)` |
| `image_conditionings_by_adding_guiding_latent` | `ImageGuide` binding |
| `image_conditionings_for_chunk(..., color_space)` | `image_conditionings_for_chunk(chunk, images, video_encoder)` |
| `images[0][0]` | `images[0].path`, read before loading |

Binding rules that Desktop relies on:

- `.at(frame 0)` replaces frame 0. A later frame guides. This equals the old `combined_image_conditionings`.
- `ImageReplace` accepts only frame 0 or `1 + 8k`. It is re-checked when a chunk maps the index.
- `ImageGuide` has no range check. The IC-LoRA reference still at frame -1 must be a guide.
- `to_ltx_images` leaves `crf` unset. The checkpoint default applies, as `resolve_crf` did.
- `api_types.ImageConditioningInput` (the Desktop tuple) is unchanged. Conversion happens at the pipeline boundary only.

## Patch decisions

| Patch or fork | Decision | Reason |
| --- | --- | --- |
| `diffusion_stage_cache` | Keep | `DiffusionStage._transformer_ctx` still takes `**kwargs`. Only DFR and multi-GPU call `resident()`. Desktop never does. |
| `diffusion_interrupt` | Keep | `DiffusionStage.__call__` is unchanged. Samplers still use `tqdm` over a sized range. |
| `diffvae_decode_vram` | Keep | The 80-frame x 608 px clamp comes from a 1.4.1 hang. 1.5.0 changed the decoder, so an ablation is needed first (see "Follow-ups"). |
| `safetensors_*`, `disk_streaming_pread`, `linux_disk_streaming_backend`, `pinned_pool_fix`, `mps_sdpa_torch` | Keep | Target symbols and contracts are unchanged. |
| `natten_libnatten_gate` | Keep | `natten_available()` still reads the module flag the patch clears. |
| `distilled_keyframe_guiding` | Delete | `ImageGuide` replaces it. |
| `TiledICLoraPipeline._video_conditionings` | Keep, adapted | Re-diffed against upstream. Only `color_space` and the still helper changed. |
| `DistilledPipelineWithSkipStage2` | Keep, adapted | Re-diffed against `_prepare_run`. Same call order, `load_conditionings` replaces `resolve_crf`. |
| `DistilledA2VPipeline`, `LTXRetakePipeline` | Keep | `DiffusionStage` calls are unchanged. Retake uses `TemporalRegionMask`, which is unchanged. |

## Edge cases checked

1. **Replace at an invalid frame.** `ImageReplace` rejects frames that are not `0` or `1 + 8k`. `.at` makes frame 0 a replace only. No Desktop path passes another replace frame.
2. **Reference still at -1.** `.at(-1)` gives `ImageGuide(-1)`. The tiled pipeline still rejects any other negative index. Upstream `assert_image_frames_in_clip` still rejects a negative frame on a stage without the IC-LoRA. The existing guard in `LTXICLoraPipeline.generate` stays.
3. **Multi-keyframe frame 0.** Old code forced "guide everything". `guide_all=True` keeps that. First and last frame image-to-video keeps replace at 0.
4. **Missing or broken image.** `load_conditionings` raises `ValueError` before any model build. The handler checks the file first (`validate_image_file`, HTTP 400), so users keep the same message.
5. **EXR stills.** Desktop has no EXR flow. `load_conditionings(images)` is called without `color_space`.
6. **Window runs.** `LoadedImageConditioning.map_frame_index` keeps the decoded still. A2V window runs reuse one decode.
7. **Skip stage 2 with auto duration.** `enhance_prompt_image` is read from `images[0].path` before loading.
8. **Torch Compile.** Timestep tables are now compact tensors. The compile path needs a live check (manual plan, section D).
9. **Same seed, new output.** Output is not bit-identical. Add this to the release notes.

## Flags and options evaluated for a Settings toggle

| Upstream option | Used by Desktop pipelines | Decision |
| --- | --- | --- |
| `--spatial-upscalings 2` (DFR) | No. DFR is a separate pipeline. Desktop uses distilled pipelines. | No toggle. A toggle would have no effect. |
| `with_rowwise_fp8_gemm` | Upstream applies it only in the DFR epilogue. | No toggle yet. See below. |
| `--refine-audio` | Only `TI2VidTwoStagesPipeline`. Desktop does not use it. | No toggle. |
| Marking (VideoSeal, AudioSeal) | Opt-in extras. | No toggle. Needs a product decision. Keep off. |
| `alpha_gen` | A one-stage IC-LoRA pipeline, not a flag. Desktop already ships an `alpha-gen` recipe on the generic IC-LoRA pipeline. | No change. |
| `--chunked`, `--compile`, `--offload` | Already mapped to Desktop behaviour or settings. | Existing "Performance (CUDA)" group stays as is. |

Why no rowwise FP8 toggle in this PR:

- It changes the numbers of every transformer pass. It is lossy (3 mantissa bits for activations).
- It needs five pipelines (fast, IC-LoRA, A2V, retake, extend) to rebuild their quantization policy. None of that can run in CI.
- This PR must stay at zero regressions. A GPU-only, unmeasured, lossy option does not fit.
- The grouping already exists. `CudaPerformanceSection` ("Performance (CUDA)") holds Torch Compile and Diffusion stage cache. A new toggle belongs there.

Proposed experiment after the upgrade ships:

1. Add `use_rowwise_fp8_gemm` to `AppSettings` (default off). Hide it unless `supports_fp8_gemm(device)`.
2. In each pipeline constructor, wrap the fp8 policy with `with_rowwise_fp8_gemm` when the setting is on.
3. Measure speed and compare outputs on an RTX 4090 and 5090 with `performance_runner` before any default change.

## Packaging

- `uv.lock` changes, so the Python deps hash changes. Windows and Linux users download a new Python environment on first start.
- Build and upload the archive for the new hash before the release (`electron/python-setup.ts`, `scripts/prepare-python.*`).
- `uv lock` fails in the current CI image with a resolver error on Python 3.14 markers. This also happens on `main`. The lock was edited by hand (only the two ltx entries). `uv lock --check` passes.
- Developers run `uv sync` in `backend/` after they pull.

## Open PRs

No open PR conflicts with the 1.5.0 change as text. Each PR got a comment with the steps to follow after the merge.

| PR | Effect |
| --- | --- |
| #464 | `pyproject.toml` and `uv.lock` merge cleanly. Run `uv lock --check` after the merge. |
| #463, #441, #446 | Budget constants and docs say "measured on 1.4.1". Re-check them against 1.5.0. #441 and #446 also conflict with `main` already. |
| #454 | `generate_still` calls `_run_inference(images=[])`. It still works. `_run_inference` has a new keyword `guide_all_images` with a default. |
| #457, #448 | New stills or recipes must go through `to_ltx_images`. They must not build upstream inputs by hand. |

## Follow-ups (not in this PR)

- Ablation of the `diffvae_decode_vram` clamp on a 32 GB card.
- Re-measure hardware budgets on 1.5.0. Refit only on an unsafe miss (`docs/hardware-budgets.md`).
- Rowwise FP8 experiment (above).
- Optional: compare the upstream `alpha_gen` pipeline with the `alpha-gen` recipe.
