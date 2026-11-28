# NAG negative prompt: Phase 0 findings

Base: branch `cursor/ltx2-1-5-0-upgrade-f80d` (PR #465, LTX-2 v1.5.0).
Source read: `Lightricks/LTX-2` tag `v1.5.0` (`ltx-core`, `ltx-pipelines`).
Plan: `nag-negative-prompt-poc-plan.md` (Bar Galili).

Status: read-only exploration. No product code changed.

## 1. Generation path (fast pipeline)

1. `handlers/video_generation_handler.py` calls `pipeline_state.pipeline.generate(...)` (line ~850).
2. `LTXFastVideoPipeline.generate` (`services/fast_video_pipeline/ltx_fast_video_pipeline.py`) calls `_run_inference`.
3. `_run_inference` calls `DistilledPipelineWithSkipStage2` (Desktop subclass in `distilled_skip_stage_2.py`) as `__call__` or `stream_chunks`.
4. The parent class is `ltx_pipelines.distilled.DistilledPipeline`. Desktop wraps and subclasses it. Desktop does not vendor it.
5. `DistilledPipeline._prepare_run` encodes the prompt with `self.prompt_encoder([prompt], ...)`. It returns `ctx_p.video_encoding` and `ctx_p.audio_encoding`.
6. `stream_chunks` passes both encodings into `generate_uniform_chunks(context=VideoAudio(...))`. `denoise_chunks` then runs stage 1 and stage 2 through `DiffusionStage`.
7. `ltx_pipelines/utils/helpers.py` (line ~540) builds `Modality(context=..., context_mask=None)`.
8. `TransformerArgsPreprocessor` (`ltx_core/model/transformer/transformer_args.py`) puts `context` into `TransformerArgs`.
9. `BasicAVTransformerBlock.forward` reads `video.context` and `audio.context`.

Result: long-form windows and both stages share one `context` per chunk. A negative context has the same lifetime.

## 2. Text cross-attention modules (v1.5.0)

All paths are under `ltx_core/model/transformer/`.

| Modality | Module | Q | K/V | Call site |
|---|---|---|---|---|
| Video | `BasicAVTransformerBlock.attn2` | video `vx` | text `video.context` | `transformer.py` ~240 |
| Audio | `BasicAVTransformerBlock.audio_attn2` | audio `ax` | text `audio.context` | `transformer.py` ~270 |

Do not patch these modules:

- `attn1` and `audio_attn1` (self-attention).
- `audio_to_video_attn` and `video_to_audio_attn` (audio and video cross-attention).

Both text cross-attention calls go through `BasicAVTransformerBlock._apply_text_cross_attention`. It has two branches:

- `cross_attention_adaln` is false: `attn(x_normed, context=context, mask=context_mask)`.
- `cross_attention_adaln` is true: `apply_cross_attention_adaln(...)`.

## 3. Finding that changes the plan: AdaLN on K/V

With `cross_attention_adaln`, the context is modulated before attention:

```
encoder_hidden_states = context * (1 + scale_kv) + shift_kv
```

`shift_kv` and `scale_kv` come from `prompt_scale_shift_table`. They also come from `prompt_timestep` when `use_prompt_adaln_single` is true.

- The negative context must use the same modulation as the positive context. If it does not, `Z_neg` is wrong.
- The query side (`attn_input`) and the output gate (`q_gate`) are shared. `q_gate` multiplies the final output.
- Open question: apply NAG before or after `q_gate`? NAG on the raw attention output (before `q_gate`) matches the plan formula. Check against KJNodes.

The `attn(...)` call also runs `to_out` (and optional per-head gating) inside `Attention.forward`. The plan formula works on the output after `V`. Decide whether NAG works before or after `to_out`:

- Before `to_out`: matches the paper. It needs a patch inside `Attention.forward`, because `out` is local there.
- After `to_out`: `to_out` is linear, so extrapolation is equivalent. The L1 norm and the clip are not equivalent.
- Per-head gating (`to_gate_logits`) depends only on `x`. It applies to both branches equally.

Open question for the PoC: use the pre-`to_out` tensor. Confirm with KJNodes.

## 4. Attention kernel and hooks

- `Attention.forward` calls `self.attention_function(q, k, v, heads)` when `mask is None`. It calls `self.masked_attention_function(...)` otherwise.
- The kernels are `PytorchAttention` (SDPA), `FlashAttention3`, `FlashAttention4`, `MPSSdpaAttention`. They come from `AttentionOps`.
- There is no processor or hook mechanism. Plan a forward patch.
- The text context mask is always `None` in `ltx-pipelines` (`context_mask=None`). The plan item "handle the negative attention mask" reduces to: pass `mask=None` for both branches. Gemma padding is handled inside the embeddings processor.
- Both contexts have the same sequence length, because the embeddings processor pads to a fixed length. Verify this once on a real run.

## 5. Text encoding

- `PromptEncoder.__call__(prompts: list[str], ...)` returns one `EmbeddingsProcessorOutput` per prompt. Gemma is loaded once, then freed.
- `ltx_retake_pipeline.py` already encodes `[prompt, negative_prompt]` in non-distilled mode. Use the same pattern: `prompts_to_encode = [prompt, negative]`.
- The prompt enhancer is controlled by `enhance_first_prompt`. It applies to the first prompt only. The negative prompt stays out of it by position.
- Desktop also pre-warms text encoding through `self._text.prepare_text_encoding(...)` in the handlers. Check that this path can include the negative prompt.

## 6. Optimizations that interact with NAG

| Feature | Where | Effect |
|---|---|---|
| FP8 cast policy | `build_fp8_cast_policy`, `device_supports_fp8` | Extra `to_k`, `to_v` projections on the negative context run through the same weights. No new weights. |
| `torch.compile` | `compile_transformer` uses `CompilationConfig()`; `ltx_core/.../compiling.py` | `mark_dynamic` covers `args.context_mask` dim 2 only. A new tensor or a changed context length can cause a recompile. Pad the negative context to the positive length. |
| CUDA graph capture | `CompilationConfig` capture is off in Desktop | No change now. A future capture would need a fixed NAG shape. |
| Streaming or offload | `OffloadMode`, `streaming_prefetch_count` | No weights added. The extra memory is small (one K/V pair per layer, transient). |
| Transformer cache patch | `services/patches/diffusion_stage_cache.py` | Caches the built transformer. A forward patch applied to the class (not the instance) survives the cache. |
| Step caching | None found | n/a |

## 7. Chosen injection point

Patch `BasicAVTransformerBlock._apply_text_cross_attention` (and, for the AdaLN branch, the `apply_cross_attention_adaln` path) in a new `services/patches/nag_cross_attention.py`. Follow the style of the other files in `services/patches/`.

- The negative context travels in a context variable that the patch reads. The pipeline wrapper sets it around one generation.
- This avoids editing `TransformerArgs`, `Modality` and `ModalitySpec`, which are upstream types.
- A strict no-op: when the context variable is unset, call the original method. The output stays bit-identical.
- Add the patch to the `services.patches` re-verify list (see `.cursor/rules/ltx2-patch-reverify.mdc`).

Alternative (rejected for the PoC): add `negative_context` to `Modality`. It is cleaner but touches many upstream call sites and breaks on every upgrade.

## 8. Data flow

```
negativePrompt (request)
  -> handler passes it to pipeline.generate(negative_prompt, nag_*)
  -> DistilledPipelineWithSkipStage2._prepare_run: prompt_encoder([prompt, negative])
  -> ctx_p (as today) + ctx_n
  -> set NAG context var (v_neg, a_neg, scale, alpha, tau)
  -> denoise_chunks stage 1 (and stage 2, per experiment)
  -> patched text cross-attention reads the context var
  -> clear the context var in a finally block
```

Notes:

- `ctx_n` is encoded once per job. It is reused across steps, both stages and all long-form windows.
- The skip-stage-2 path (`distilled_skip_stage_2.py`) has its own `_prepare_run` copy and must be changed too.
- The A2V pipeline (`ltx_a2v_pipeline.py`) already takes a `negative_prompt`. Check what it does with it in distilled mode before reusing the name.

## 9. Affected callers

Wire the new parameters for each caller, or state "not supported yet":

- `handlers/video_generation_handler.py`: fast text-to-video, image-to-video, keyframes (line ~850).
- Long-form windows: `windowed_run.py` and `stream_chunks` (same context var).
- `distilled_skip_stage_2.py`: the skip-stage-2 variant.
- `handlers/retake_handler.py`, `services/features/video/retake.py`, `extend.py`, `edit_executor.py`: they pass `default_negative_prompt`. Keep NAG off for them in the PoC.
- `handlers/ic_lora_handler.py`: IC-LoRA (`ic_lora_pipeline`, tiled variant). Out of scope for the PoC.
- A2V (`a2v_pipeline`): out of scope for the PoC.
- Remote API: `remote/dto.py` has no negative prompt field. Phase 3.
- Seed Hunt drafts and rerenders: Phase 3 (persist NAG parameters).

Existing wiring to know:

- `api_types.py` already has `negativePrompt: str = ""` on the video request (line ~831). The fast pipeline ignores it today.
- `video_generation_handler.py` falls back to `default_negative_prompt` only on the A2V path (line ~968).
- `ltx2_server.py` defines `DEFAULT_NEGATIVE_PROMPT` (line ~318).

## 10. Open questions (answered)

1. Extrapolation convention: KJNodes `LTX2_NAG` uses `s * Z_pos - (s - 1) * Z_neg`. Desktop uses the same, so the KJNodes defaults (`scale=11`, `alpha=0.25`, `tau=2.5`) apply as they are. In the plan's formula this is `s_plan = scale - 1`.
2. Guided tensor: the attention output after `V`, before the per-head gate and `to_out`. KJNodes does the same (`ltxv_crossattn_forward_nag`). The gate and `to_out` stay shared.
3. Clip: KJNodes clips when the L1 ratio is above `tau` and maps a NaN ratio to 10. Desktop copies both.
4. KJNodes gives the NAG context no mask. Desktop gives the negative half no mask either.
5. `negativePrompt` contract: the field already exists. With the flag off, it stays ignored on the fast path. With the flag on, a non-empty value turns NAG on.
6. `DEFAULT_NEGATIVE_PROMPT` is not used for NAG. It was written for CFG models. A hidden default list is a Phase 3 decision.
7. Stage 1 only or both stages: Phase 1 guides every denoise call (both stages). The sweep belongs to Phase 2.

New finding (API text encoding): `LTXTextEncoder` patches `PromptEncoder.__call__` in API mode. It returns zeros for every prompt after the first. NAG with a zero negative is wrong. The handler skips NAG when the text encoding is not local, and logs a warning.

## 11. Implementation status (Phases 1-3)

Done on branch `nag-negative-prompt-poc`:

- `services/fast_video_pipeline/nag_config.py`: `NagConfig`, `nag_config_for_request` (enabled when `negativePrompt` is non-empty and `scale > 0`, unless `LTX_NAG_ENABLED=0`), optional `LTX_NAG_SCALE`, `LTX_NAG_ALPHA`, `LTX_NAG_TAU`.
- `services/patches/nag_cross_attention.py`: the `attn2` patch and `nag_scope`.
- `services/fast_video_pipeline/nag_distilled.py`: encodes the negative prompt in the same Gemma pass, appends it to the video context, and scopes NAG to the denoise calls.
- `DistilledPipelineWithSkipStage2`, `LTXFastVideoPipeline.generate`, the protocol, the fake and `video_generation_handler.py` pass `nag` through.
- `state/app_settings.py`, `AppSettingsContext.tsx`, `SettingsModal.tsx`: `nag_scale` (`nagScale`, default `11.0`) setting in **General > Generation**.
- `api_types.py`, `queued_generation_handler.py`, `remote/dto.py`: `nagScale`, `nagAlpha`, `nagTau` persisted on queued generations and projected to Remote so retries and rerenders reproduce.
- `textToVideo.ts`, `imageToVideo.ts`, `loraRecipe.ts`: `Negative prompt` textarea exposed on Home and Remote forms.
- Tests: `backend/tests/test_nag_cross_attention.py`, `backend/tests/test_remote_generations.py`, `textToVideo.test.ts`, `imageToVideo.test.ts`.

