# Remote (Gemini) prompt enhancement — design

**Date:** 2026-07-21
**Branch:** `feature/local-lora-aware-prompt-enhancer`
**Status:** Approved, implementing.

## Problem

The catalog-aware Enhance button (`POST /api/enhance-prompt`) only runs the local Gemma text
encoder (`LtxPromptEnhancerPipeline`). It's hidden unless a local Gemma checkpoint is downloaded
(`GenSpace.tsx` `hasLocalTextEncoder` gate), which means:

- Users doing local video generation who haven't downloaded Gemma (or don't want to spend the
  VRAM/load time on it) can't use Enhance at all, even though generation itself is local.
- There's no way to get the catalog-aware rewrite without paying the local-Gemma cost.

Two remote alternatives were investigated and ruled out:

- **`ltxv-api`'s draft `/v1/prompt-enhance`** (branch `add-prompt-enhance-endpoint`, unmerged,
  2 weeks stale, org-gated to `Lightricks` only, no context-override param yet) — cross-repo
  dependency, out of scope for this change.
- **FAL's `any-llm`** — FAL's only generic text-in/text-out LLM proxy is deprecated/unsupported;
  FAL today is media-generation only (image/video/audio/3D), no text models.

## Solution

Add a second `PromptEnhancerPipeline`-shaped implementation that calls Gemini's
`generateContent` REST API directly, reusing the exact pattern already proven in this codebase at
`backend/handlers/suggest_gap_prompt_handler.py:142-164` (`systemInstruction` for the system
prompt, `contents` for the user prompt + optional inline image, `x-goog-api-key` auth). This uses
the `gemini_api_key` setting that already exists (`AppSettings.gemini_api_key`,
`SettingsResponse.has_gemini_api_key` / frontend `hasGeminiApiKey`) — no new backend infra, no
cross-repo dependency.

The Enhance button gets a provider choice: **Local** (unchanged) or **API** (new, Gemini-backed).
The catalog-aware system-prompt builders (`system_prompt.py`) are reused byte-for-byte across both
— only the "call a model" step differs.

## Backend

### `EnhancePromptRequest` (`backend/api_types.py`)

Add `provider: Literal["local", "api"] = "local"`. Default preserves current behavior exactly for
every existing caller.

### `GeminiPromptEnhancerPipeline` (new: `backend/services/prompt_enhancer_pipeline/gemini_prompt_enhancer_pipeline.py`)

A plain class, constructed once with `http: HTTPClient` (the same instance already threaded
through `AppHandler.__init__`, no `ServiceBundle` change needed — mirrors how
`SuggestGapPromptHandler` receives `http` directly rather than through the bundle). Not shaped
like the local `PromptEnhancerPipeline` Protocol's `.create(gemma_root, device)` factory, since
construction doesn't depend on a checkpoint path — instead it's a long-lived instance whose calls
take the API key as an argument (the key can change at runtime via Settings):

```python
class GeminiPromptEnhancerPipeline:
    def __init__(self, http: HTTPClient) -> None: ...
    def enhance_t2v(self, prompt: str, system_prompt: str | None, seed: int, *, api_key: str) -> str: ...
    def enhance_i2v(self, prompt: str, image_path: str, system_prompt: str | None, seed: int, *, api_key: str) -> str: ...
```

Request shape mirrors `suggest_gap_prompt_handler.py`: POST to
`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`,
`systemInstruction.parts[0].text = system_prompt` (omitted if `None`), `contents[0].parts` =
`[{"text": prompt}]` plus, for `enhance_i2v`, an appended `{"inlineData": {"mimeType", "data"}}`
part (image read + base64-encoded, same helper shape as `_read_image_file_as_base64`).
`generationConfig.seed = seed` (Gemini API supports a best-effort seed). Response text extracted
via the same `candidates[0].content.parts[0].text` shape as `_extract_gemini_text` — factor that
helper out of `suggest_gap_prompt_handler.py` into a shared spot (e.g. `services/prompt_enhancement/`
or a small `gemini_client` util) so both call sites use one parser. Missing/failed key → the
pipeline lets non-200 responses raise; the handler maps to `HTTPError`.

### `PromptEnhancementHandler` (`backend/handlers/prompt_enhancement_handler.py`)

- New constructor param `gemini_pipeline: GeminiPromptEnhancerPipeline` (concrete type is fine —
  no Protocol needed for a single implementation with no local/test-swap requirement beyond
  `FakeHTTPClient`, which already covers the test boundary).
- `enhance()`: branch the precondition check on `req.provider` instead of unconditionally requiring
  `gemma_root`:
  - `provider == "local"` → today's check (`resolve_gemma_root()` or `409
    LOCAL_TEXT_ENCODER_NOT_AVAILABLE`).
  - `provider == "api"` → check `self.state.app_settings.gemini_api_key`; empty →
    `400 GEMINI_API_KEY_MISSING`.
- `_run_free_rewrite` / `_run_template_fill`: branch on `req.provider` to call either
  `self._load_prompt_enhancer_pipeline(gemma_root)` (local, unchanged, still calls
  `evict_gpu_pipeline_for_prompt_enhancement()`) or `self._gemini_pipeline.enhance_t2v/i2v(...,
  api_key=self.state.app_settings.gemini_api_key)` (API — no GPU eviction, nothing to evict).
- **Logging**: one `logger.info` at the top of each dispatch — `"Enhancing prompt via local Gemma"`
  or `"Enhancing prompt via Gemini API"` — so it's visible which path served a given request.

### Wiring (`backend/app_handler.py`)

Construct `GeminiPromptEnhancerPipeline(http)` directly in `AppHandler.__init__` (using the `http`
param already passed in) and pass it into `PromptEnhancementHandler(..., gemini_pipeline=...)`.
No `ServiceBundle` field, no new constructor param threaded through `build_default_service_bundle`
— the instance depends on nothing test-swappable beyond `http`, which is already a `ServiceBundle`
field.

### Tests

- `tests/test_prompt_enhancement.py`: new cases for `provider="api"` — success path (queue a
  `FakeResponse` on `fake_services.http` shaped like Gemini's real response), missing-key 400,
  non-200 upstream error mapping. Existing `provider` omission still defaults to `"local"` and all
  current tests pass unchanged.
- No new Fake class needed — `FakeHTTPClient.queue("post", FakeResponse(...))` is the full test
  seam, consistent with how `suggest_gap_prompt_handler` itself is tested (if it has coverage) or
  how other `http`-backed services are tested in this suite.

## Frontend

### Button visibility (`GenSpace.tsx`)

`canEnhancePrompt` gate becomes `isLocalMode && (hasLocalTextEncoder || hasGeminiApiKey) && (mode
=== 'video' || mode === 'ic-lora') && prompt.trim().length > 0 && !isGenerationInProgressForEnhance`.

### Provider toggle

- New state `enhanceProvider: 'local' | 'api'`, defaulting to `'local'` if `hasLocalTextEncoder`,
  else `'api'` (so it always defaults to whichever is actually usable).
- Toggle UI shown only when **both** `hasLocalTextEncoder` and `hasGeminiApiKey` are true;
  otherwise no toggle — the button silently uses whichever single option is available (matches
  today's zero-toggle behavior for users with only one option).
- Button label: `Enhance` when `enhanceProvider === 'local'`, `Enhance (API)` when `'api'`.
- `runEnhance` passes `provider: enhanceProvider` into the `ApiClient.enhancePrompt(...)` call.

### Regenerate OpenAPI types

`pnpm openapi:generate` after the `api_types.py` change, to pick up `provider` in
`EnhancePromptRequest` for the generated TS client.

## Out of scope

- Negative-prompt / scene-setting fields — not part of this endpoint's response shape at all
  (Gemini's `generateContent` just returns text); no change needed here.
- The `ltxv-api` draft endpoint and FAL — both ruled out during research, documented above for
  posterity, not revisited unless the Gemini path proves insufficient.
- The legacy `prompt_enhancer_enabled_t2v`/`_i2v` toggle (remote API-mode auto-enhance) is untouched
  — it remains a separate, unrelated feature.
