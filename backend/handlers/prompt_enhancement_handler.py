"""Local, catalog-aware prompt enhancement handler."""

from __future__ import annotations

import logging
import random
import uuid
from threading import RLock
from typing import TYPE_CHECKING, Literal, NamedTuple

from _routes._errors import HTTPError
from api_types import (
    ConditioningType,
    EnhancePromptRequest,
    EnhancePromptResponse,
    IcLoraCatalogItem,
    LoraCatalogItem,
    LTXLocalModelId,
    PromptEnhancerStatusResponse,
    PromptProvenance,
)
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from server_utils.media_validation import normalize_optional_path, validate_image_file
from services.analytics import resolution_for_provenance as shared_resolution_for_provenance
from services.analytics import prompt_provenance_label as shared_prompt_provenance_label
from services.gemini_text_client import resolve_gemini_model
from services.generation_interrupt import is_cancel_exception
from services.interfaces import PromptEnhancerPipeline
from services.lora_catalog import LoraCatalogProvider
from services.prompt_enhancement import (
    build_audio_visual_caption_system_prompt,
    build_conditioning_system_prompt,
    build_ic_lora_enhancement_system_prompt,
    build_image_edit_system_prompt,
    build_image_generation_system_prompt,
    build_keyframe_enhancement_system_prompt,
    build_lora_enhancement_system_prompt,
    build_template_fill_system_prompt,
    enforce_trigger_placements,
    fill_prompt_template,
    parse_template_fill_response,
)
from services.prompt_enhancement.i2v_frames import KeyframeStill
from services.prompt_enhancer_pipeline.gemini_prompt_enhancer_pipeline import GeminiPromptEnhancerPipeline
from services.services_utils import get_device_type
from state.app_state_types import AppState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

# Manual Enhance is exploratory — a redo should offer a different rewrite even when generation
# seeds are locked — so it always draws fresh here rather than going through
# StateHandlerBase._resolve_seed(). Automatic enhancement is the opposite case; see
# _automatic_enhancement_seed().
_MAX_ENHANCE_SEED = 2147483647


class ResolvedGenerationPrompt(NamedTuple):
    """The prompt a local generation should submit, and how it got (or will get) enhanced."""

    prompt: str
    enhance_via_api: bool
    enhanced_locally: bool
    skip_recipe_wrap: bool = False

    @property
    def enhancement(self) -> str:
        """Where the rewrite happens, for run logs: local, server-side, or nowhere."""
        if self.enhanced_locally:
            return "local"
        return "api" if self.enhance_via_api else "none"


PromptProvenanceLabel = Literal["raw", "manually-enhanced", "auto-enhanced"]


def prompt_provenance_label(
    provenance: PromptProvenance,
    resolved: ResolvedGenerationPrompt,
) -> PromptProvenanceLabel:
    """How this run asked for the prompt to be passed. Not the prompt text."""
    return shared_prompt_provenance_label(
        provenance,
        enhanced_locally=resolved.enhanced_locally,
        enhance_via_api=resolved.enhance_via_api,
    )


class PromptEnhancementHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        lora_catalog_provider: LoraCatalogProvider,
        prompt_enhancer_pipeline_class: type[PromptEnhancerPipeline],
        gemini_pipeline: GeminiPromptEnhancerPipeline,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text_handler = text_handler
        self._lora_catalog_provider = lora_catalog_provider
        self._prompt_enhancer_pipeline_class = prompt_enhancer_pipeline_class
        self._gemini_pipeline = gemini_pipeline

    def _random_seed(self) -> int:
        return random.randint(0, _MAX_ENHANCE_SEED)

    def status(self, *, prefer_gemini: bool) -> PromptEnhancerStatusResponse:
        """Flags for the Home Enhance control. ``prefer_gemini`` is Remote's default."""
        settings = self.state.app_settings
        has_gemini = bool(settings.gemini_api_key.strip())
        local_downloaded = (
            self._text_handler.resolve_prompt_enhancer_root_if_downloaded() is not None
        )
        has_local = local_downloaded and not self.config.force_api_generations
        if prefer_gemini:
            default_provider: Literal["local", "api"] = (
                "api" if has_gemini else "local" if has_local else "api"
            )
        else:
            preference = settings.prompt_enhancer_provider_preference
            if preference == "api":
                preferred: Literal["local", "api"] = "api"
            elif preference == "local" and has_local:
                preferred = "local"
            elif has_local:
                preferred = "local"
            else:
                preferred = "api"
            default_provider = (
                "local"
                if preferred == "api" and not has_gemini and has_local
                else preferred
            )
        return PromptEnhancerStatusResponse(
            exploreAutoEnhancePrompts=settings.explore_auto_enhance_prompts,
            hasGeminiApiKey=has_gemini,
            localEnhancementSupported=has_local,
            defaultProvider=default_provider,
            canToggleProvider=has_local and has_gemini,
            showManualEnhance=not settings.explore_auto_enhance_prompts,
        )

    def _automatic_enhancement_seed(
        self, generation_seed: int, *, explicit_generation_seed: bool
    ) -> int:
        """The seed an automatic rewrite runs on, given the seed the generation resolved to.

        Automatic enhancement is part of what produces the video: the model never sees the
        prompt as typed, only the rewrite. Drawing that rewrite independently would leave a
        pinned local generation irreproducible — same seed, different prompt, different clip —
        so an explicit request seed or seed lock makes the rewrite ride the generation's seed.
        An unlocked request without an explicit seed draws fresh like manual Enhance does.
        """
        if explicit_generation_seed or self.state.app_settings.seed_locked:
            return generation_seed
        return self._random_seed()

    def enhance(self, req: EnhancePromptRequest) -> EnhancePromptResponse:
        # Enhance never occupies the GPU slot (see PipelinesHandler.
        # evict_gpu_pipeline_for_prompt_enhancement) but still needs to mutually exclude with
        # generation and with itself — an abandoned/orphaned enhance call (e.g. the tab reloaded
        # mid-request) must not race a Generate click, a second Enhance click, or a generation
        # that's still loading its pipeline (reserved_generation_start covers that window; a bare
        # is_generation_running() check does not — see its own docstring). The "api" generation
        # slot gives us the mutual exclusion for free: it's the same bookkeeping every other
        # handler already does, and it doesn't require gpu_slot to be set.
        with self._generation.reserved_generation_start():
            gemma_root: str | None = None
            if req.provider == "local":
                gemma_root = self._text_handler.resolve_prompt_enhancer_root_if_downloaded()
                if gemma_root is None:
                    raise HTTPError(409, "LOCAL_TEXT_ENCODER_NOT_AVAILABLE")
            elif not self.state.app_settings.gemini_api_key.strip():
                raise HTTPError(400, "GEMINI_API_KEY_MISSING")

            generation_id = uuid.uuid4().hex[:8]
            self._generation.start_api_generation(generation_id)
            try:
                enhanced = self._resolve_and_enhance(req, gemma_root)
            except HTTPError as e:
                self._generation.fail_generation(e.detail)
                raise
            except Exception as e:
                self._generation.fail_generation(str(e))
                raise HTTPError(500, str(e)) from e

            self._generation.complete_generation(enhanced)
            return EnhancePromptResponse(enhancedPrompt=enhanced)

    def _resolve_and_enhance(self, req: EnhancePromptRequest, gemma_root: str | None) -> str:
        if req.mediaType == "image":
            # No catalog LoRA concept for images (validated at the request level) — always an
            # explicit, image-domain system prompt, never the video-oriented generic fallback.
            system_prompt = (
                build_image_edit_system_prompt() if req.imagePath is not None
                else build_image_generation_system_prompt()
            )
            return self._run_free_rewrite(req, system_prompt, gemma_root)

        if req.icLoraId is not None:
            ic_lora = self._lora_catalog_provider.get_ic_lora(req.icLoraId)
            if ic_lora is None:
                raise HTTPError(404, "LORA_CATALOG_ID_NOT_FOUND")
            return self._enhance_ic_lora(ic_lora, req, gemma_root)

        if req.loraCatalogIds:
            loras: list[LoraCatalogItem] = []
            for catalog_id in req.loraCatalogIds:
                lora = self._lora_catalog_provider.get_lora(catalog_id)
                if lora is None:
                    raise HTTPError(404, "LORA_CATALOG_ID_NOT_FOUND")
                loras.append(lora)
            return self._enhance_loras(loras, req, gemma_root)

        if req.conditioningType is not None:
            system_prompt = build_conditioning_system_prompt(req.conditioningType)
            return self._run_free_rewrite(req, system_prompt, gemma_root)

        return self._run_free_rewrite(req, self._default_video_system_prompt(req), gemma_root)

    def _default_video_system_prompt(self, req: EnhancePromptRequest) -> str | None:
        if req.keyframes:
            return self._keyframe_system_prompt()
        return self._video_system_prompt(t2v=req.imagePath is None)

    def _keyframe_system_prompt(self, model_id: LTXLocalModelId | None = None) -> str:
        spec = self._text_handler.ltx_model_spec(model_id)
        audio_visual = spec is not None and spec.wants_audio_visual_captions
        return build_keyframe_enhancement_system_prompt(audio_visual=audio_visual)

    def _video_system_prompt(
        self, *, t2v: bool, model_id: LTXLocalModelId | None = None
    ) -> str | None:
        """The model's own caption style, or None to keep each provider's default.

        Only the audio-visual generations (2.5) need this: their captions cover the soundscape,
        which neither the generic Gemini fallback nor a 2.3-era prompt asks for.
        """
        spec = self._text_handler.ltx_model_spec(model_id)
        if spec is None or not spec.wants_audio_visual_captions:
            return None
        return build_audio_visual_caption_system_prompt(t2v=t2v)

    def resolve_for_generation(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        generation_seed: int,
        explicit_generation_seed: bool = False,
        image_path: str | None = None,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
        ic_lora: IcLoraCatalogItem | None = None,
        lora_catalog_ids: list[str] | None = None,
        conditioning_type: ConditioningType | None = None,
        local_model_id: LTXLocalModelId | None = None,
        explore_generation: bool = False,
        audio_path: str | None = None,
    ) -> ResolvedGenerationPrompt:
        """The single enhancement policy for prompt-bearing local generation.

        Local encoding attempts a rewrite unless this is an Explore generation with
        auto-enhance off. API encoding forwards ``enhance_prompt`` according to the T2V/I2V
        settings (I2V and keyframe jobs use the I2V toggle; T2V, IC-LoRA, and A2V —
        including A2V with an optional start frame — use T2V). Three things still suppress
        both paths: an "enhanced" provenance, meaning the
        user already ran Enhance by hand and kept the result (rewriting a rewrite drifts the
        prompt away from what they approved); an empty prompt, which has nothing to expand; and
        Explore auto-enhance off, which sends the prompt as typed (Gen Space is unchanged).

        The two text-encoding paths enhance in different places: API encoding rewrites
        server-side inside the same /prompt-embedding call, so it only needs the flag forwarded.
        Local encoding has no such step, so the rewrite happens here — without it the model sees
        the prompt as typed, which for a version captioned in 150-220 word audio-visual
        paragraphs (2.5) lands far outside its training distribution and it invents the rest.

        ``lora_catalog_ids`` are the catalog ids of the plain LoRAs the request selected. They
        get the same catalog-aware treatment manual Enhance gives them, because the generic
        rewrite would paraphrase a required trigger phrase away and leave the adapter inert.

        ``generation_seed`` is the seed the caller already resolved for this run (an explicit
        request seed, the lock, or a fresh draw). It reaches the rewrite when
        ``explicit_generation_seed`` is true or seed locking is enabled — see
        _automatic_enhancement_seed() — and never reaches the API path, whose ``enhance_prompt``
        flag takes no seed.

        Must be called before the pipeline is loaded and before start_generation: the enhancer
        needs the VRAM a resident pipeline holds, and PipelinesHandler refuses to evict one
        while a generation is running.
        """
        preview = self._preview_resolution(
            prompt,
            provenance=provenance,
            explore_generation=explore_generation,
            local_model_id=local_model_id,
        )
        if preview is not None:
            if preview.skip_recipe_wrap and provenance != "enhanced" and prompt.strip():
                logger.info(
                    "Explore auto-enhance off: sending prompt as typed (%d chars)",
                    len(prompt),
                )
            return preview
        enhanced, rewritten = self.enhance_for_generation(
            prompt,
            generation_seed=generation_seed,
            explicit_generation_seed=explicit_generation_seed,
            image_path=image_path,
            last_image_path=last_image_path,
            keyframes=keyframes,
            duration=duration,
            fps=fps,
            ic_lora=ic_lora,
            loras=self._resolve_generation_loras(lora_catalog_ids),
            conditioning_type=conditioning_type,
            local_model_id=local_model_id,
        )
        return ResolvedGenerationPrompt(enhanced, False, rewritten, False)

    def preview_prompt_provenance(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        explore_generation: bool = False,
        local_model_id: LTXLocalModelId | None = None,
        force_api: bool = False,
    ) -> PromptProvenanceLabel:
        """The analytics label for this request, without rewriting the prompt.

        A local rewrite that has not run yet counts as auto-enhanced. That is the
        mode this run requested; a later Gemma fallback still used that mode.
        """
        preview = self._preview_resolution(
            prompt,
            provenance=provenance,
            explore_generation=explore_generation,
            local_model_id=local_model_id,
            force_api=force_api,
        )
        if preview is None:
            preview = ResolvedGenerationPrompt(prompt, False, True)
        return shared_prompt_provenance_label(
            provenance,
            enhanced_locally=preview.enhanced_locally,
            enhance_via_api=preview.enhance_via_api,
        )

    def _preview_resolution(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        explore_generation: bool,
        local_model_id: LTXLocalModelId | None,
        force_api: bool = False,
    ) -> ResolvedGenerationPrompt | None:
        """Resolution when no local rewrite is required.

        None means a local rewrite will be attempted.
        """
        decided = shared_resolution_for_provenance(
            provenance,
            prompt=prompt,
            explore_generation=explore_generation,
            explore_auto_enhance_prompts=self.state.app_settings.explore_auto_enhance_prompts,
            use_local_encoding=self._text_handler.should_use_local_encoding(local_model_id),
            prompt_enhancer_enabled=self.state.app_settings.prompt_enhancer_enabled,
            local_enhancer_available=(
                self._text_handler.resolve_prompt_enhancer_root_if_downloaded(local_model_id)
                is not None
            ),
            force_api=force_api,
        )
        if decided is None:
            return None
        enhance_via_api, enhanced_locally, skip_recipe_wrap = decided
        return ResolvedGenerationPrompt(prompt, enhance_via_api, enhanced_locally, skip_recipe_wrap)

    def api_enhance_prompt(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        audio_path: str | None = None,
    ) -> bool:
        """Whether the LTX API should rewrite this prompt on the server.

        Used for API text encoding (``enhance_prompt`` on /prompt-embedding) and for
        fully-remote video (``enhance_prompt`` on /v2/text-to-video and siblings). The
        LTX gateway defaults that flag to true when omitted, so callers must send it.

        One setting covers every conditioning: the conditioning arguments stay in the
        signature because callers pass them, but they no longer select between gates.
        """
        if provenance == "enhanced" or not prompt.strip():
            return False
        return self.state.app_settings.prompt_enhancer_enabled

    def _resolve_generation_loras(self, catalog_ids: list[str] | None) -> list[LoraCatalogItem]:
        """The catalog entries behind a generation's selected plain LoRAs, by id alone.

        Unlike manual Enhance, an unresolvable id doesn't 404: automatic enhancement is a
        quality step layered onto a generation the user already started, so a stale id costs
        the catalog treatment rather than the run.
        """
        items: list[LoraCatalogItem] = []
        for catalog_id in catalog_ids or []:
            lora = self._lora_catalog_provider.get_lora(catalog_id)
            if lora is None:
                logger.info("Skipping unknown LoRA catalog id for enhancement: %s", catalog_id)
                continue
            # A templated entry needs the template-fill path, which the multi-select free
            # rewrite has no shape for. No plain catalog LoRA has one today.
            if lora.prompt_template is not None:
                continue
            items.append(lora)
        return items

    def enhance_for_generation(
        self,
        prompt: str,
        *,
        generation_seed: int,
        explicit_generation_seed: bool,
        image_path: str | None,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
        ic_lora: IcLoraCatalogItem | None = None,
        loras: list[LoraCatalogItem] | None = None,
        conditioning_type: ConditioningType | None = None,
        local_model_id: LTXLocalModelId | None = None,
    ) -> tuple[str, bool]:
        """Rewrite ``prompt`` on the local enhancer: ``(prompt, rewritten)``.

        ``rewritten`` is False on every fallback below, which is what lets a caller's run log
        distinguish "the enhancer rewrote this" from "the enhancer never ran".

        Home auto-enhance calls this whenever the enhancer checkpoint is on disk, including
        when text embeddings come from the API. A plain API-encoding run with no local
        checkpoint still enhances server-side and never gets here. Deliberately not
        `enhance()` — the caller already holds the generation slot, and there's no provider
        choice to make, only "is the enhancer on disk".

        Only cancellation is re-raised. Enhancement is otherwise a quality step, so a missing
        checkpoint or failed rewrite degrades to a deterministic fallback rather than failing
        the generation.
        """
        loras = loras or []
        if not prompt.strip():
            return prompt, False
        gemma_root = self._text_handler.resolve_prompt_enhancer_root_if_downloaded(
            model_id=local_model_id
        )
        if gemma_root is None:
            logger.info("Skipping automatic enhancement: no local prompt enhancer downloaded")
            return self._generation_fallback_prompt(prompt, ic_lora, loras), False

        try:
            pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
            seed = self._automatic_enhancement_seed(
                generation_seed, explicit_generation_seed=explicit_generation_seed
            )
            if ic_lora is not None:
                enhanced = self._enhance_ic_lora_for_generation(pipeline, prompt, ic_lora, seed)
            else:
                system_prompt = self._generation_system_prompt(
                    loras=loras,
                    image_path=image_path,
                    keyframes=keyframes,
                    conditioning_type=conditioning_type,
                    local_model_id=local_model_id,
                )
                if image_path is not None or keyframes:
                    first_path = image_path or (keyframes[0][0] if keyframes else None)
                    assert first_path is not None
                    enhanced = pipeline.enhance_i2v(
                        prompt,
                        first_path,
                        system_prompt=system_prompt,
                        seed=seed,
                        last_image_path=None if keyframes else last_image_path,
                        keyframes=keyframes,
                        duration=duration,
                        fps=fps,
                    )
                else:
                    enhanced = pipeline.enhance_t2v(prompt, system_prompt=system_prompt, seed=seed)
                if enhanced.strip():
                    # Deterministic, same as manual Enhance: the rewrite is asked for the
                    # trigger phrases, never trusted to have kept them.
                    enhanced = enforce_trigger_placements(enhanced, loras)
        except Exception as exc:
            if is_cancel_exception(exc):
                raise
            logger.warning("Automatic local enhancement failed; using the fallback prompt", exc_info=True)
            return self._generation_fallback_prompt(prompt, ic_lora, loras), False

        if not enhanced.strip():
            return self._generation_fallback_prompt(prompt, ic_lora, loras), False
        logger.info(
            "Enhanced prompt locally for generation (%d -> %d chars)",
            len(prompt),
            len(enhanced),
        )
        return enhanced, True

    def _generation_fallback_prompt(
        self,
        prompt: str,
        ic_lora: IcLoraCatalogItem | None,
        loras: list[LoraCatalogItem],
    ) -> str:
        """Preserve every non-template catalog adapter's trigger when no rewrite is available."""
        items: list[LoraCatalogItem] = list(loras)
        if ic_lora is not None and ic_lora.prompt_template is None:
            items.append(ic_lora)
        return prompt if not items else enforce_trigger_placements(prompt, items)

    def _generation_system_prompt(
        self,
        *,
        loras: list[LoraCatalogItem],
        image_path: str | None,
        keyframes: list[KeyframeStill] | None,
        conditioning_type: ConditioningType | None,
        local_model_id: LTXLocalModelId | None = None,
    ) -> str | None:
        if loras:
            # Same treatment manual Enhance's multi-select path applies. It replaces the
            # model's own caption style rather than layering onto it: what the adapters need
            # from the prompt is the more specific constraint, and the LoRA block already
            # carries its own output-format instruction.
            return build_lora_enhancement_system_prompt(loras)
        if conditioning_type is not None:
            # Built-in canny/depth IC-LoRA: no catalog entry to draw from, but the same
            # "describe the reference scene faithfully" discipline manual Enhance applies.
            return build_conditioning_system_prompt(conditioning_type)
        if keyframes:
            return self._keyframe_system_prompt(model_id=local_model_id)
        return self._video_system_prompt(t2v=image_path is None, model_id=local_model_id)

    def _enhance_ic_lora_for_generation(
        self,
        pipeline: PromptEnhancerPipeline,
        prompt: str,
        ic_lora: IcLoraCatalogItem,
        seed: int,
    ) -> str:
        """The catalog-aware treatment manual Enhance gives an IC-LoRA, minus the error surface.

        Text-only on purpose, exactly like the manual IC-LoRA path: an IC-LoRA's reference is
        its driving clip (or the control video built from it), not an i2v conditioning still.
        A template that can't be filled raises out to the caller's typed-prompt fallback rather
        than failing the generation.
        """
        if ic_lora.prompt_template is not None:
            raw = pipeline.enhance_t2v(
                prompt, system_prompt=build_template_fill_system_prompt(ic_lora), seed=seed
            )
            values = parse_template_fill_response(raw, set(ic_lora.prompt_template.placeholders))
            return fill_prompt_template(ic_lora.prompt_template, values)
        enhanced = pipeline.enhance_t2v(
            prompt, system_prompt=build_ic_lora_enhancement_system_prompt(ic_lora), seed=seed
        )
        # Trigger enforcement on an empty rewrite would generate from a bare trigger word.
        if not enhanced.strip():
            return enhanced
        return enforce_trigger_placements(enhanced, [ic_lora])

    def _enhance_loras(
        self, loras: list[LoraCatalogItem], req: EnhancePromptRequest, gemma_root: str | None
    ) -> str:
        # None of the plain LoRAs in the catalog have a prompt_template today (only IC-LoRAs
        # do) — the multi-select path is always a free rewrite.
        system_prompt = build_lora_enhancement_system_prompt(loras)
        enhanced = self._run_free_rewrite(req, system_prompt, gemma_root)
        return enforce_trigger_placements(enhanced, loras)

    def _enhance_ic_lora(
        self, ic_lora: IcLoraCatalogItem, req: EnhancePromptRequest, gemma_root: str | None
    ) -> str:
        if ic_lora.prompt_template is not None:
            return self._run_template_fill(ic_lora, req, gemma_root)
        system_prompt = build_ic_lora_enhancement_system_prompt(ic_lora)
        enhanced = self._run_free_rewrite(req, system_prompt, gemma_root)
        return enforce_trigger_placements(enhanced, [ic_lora])

    def _run_free_rewrite(
        self, req: EnhancePromptRequest, system_prompt: str | None, gemma_root: str | None
    ) -> str:
        # Reject an invalid/unreadable/oversized path before it reaches either provider — the
        # API path in particular would otherwise base64-encode and ship arbitrary file bytes to
        # a third-party API with no gate at all.
        keyframes = self._validated_keyframes(req)
        image_path = (
            None if keyframes else normalize_optional_path(req.imagePath)
        )
        last_image_path = (
            None
            if req.mediaType == "image" or keyframes
            else normalize_optional_path(req.lastImagePath)
        )
        if image_path is not None:
            validate_image_file(image_path)
        if last_image_path is not None:
            validate_image_file(last_image_path)

        seed = self._random_seed()
        first_path = image_path or (keyframes[0][0] if keyframes else None)
        if req.provider == "api":
            resolved_model = resolve_gemini_model(self.state.app_settings.gemini_model)
            logger.info("Enhancing prompt via Gemini API (%s)", resolved_model)
            api_key = self.state.app_settings.gemini_api_key.strip()
            if first_path is not None:
                return self._gemini_pipeline.enhance_i2v(
                    req.prompt,
                    first_path,
                    system_prompt=system_prompt,
                    seed=seed,
                    api_key=api_key,
                    model=resolved_model,
                    last_image_path=last_image_path,
                    keyframes=keyframes,
                    duration=req.duration,
                    fps=req.fps,
                )
            return self._gemini_pipeline.enhance_t2v(
                req.prompt,
                system_prompt=system_prompt,
                seed=seed,
                api_key=api_key,
                model=resolved_model,
            )

        logger.info("Enhancing prompt via local Gemma")
        assert gemma_root is not None
        pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
        if first_path is not None:
            return pipeline.enhance_i2v(
                req.prompt,
                first_path,
                system_prompt=system_prompt,
                seed=seed,
                last_image_path=last_image_path,
                keyframes=keyframes,
                duration=req.duration,
                fps=req.fps,
            )
        return pipeline.enhance_t2v(req.prompt, system_prompt=system_prompt, seed=seed)

    def _validated_keyframes(self, req: EnhancePromptRequest) -> list[KeyframeStill] | None:
        if req.mediaType == "image" or not req.keyframes:
            return None
        frames: list[KeyframeStill] = []
        for keyframe in req.keyframes:
            path = normalize_optional_path(keyframe.imagePath)
            if path is None:
                raise HTTPError(400, "Each keyframe requires an image path")
            validate_image_file(path)
            frames.append((path, keyframe.frameIndex, keyframe.strength))
        frames.sort(key=lambda item: item[1])
        return frames

    def _run_template_fill(
        self, ic_lora: IcLoraCatalogItem, req: EnhancePromptRequest, gemma_root: str | None
    ) -> str:
        # req.imagePath is intentionally unused here — template fill is always a text-only
        # enhance_t2v call (the fixed template scaffold carries no reference-image slot), unlike
        # the free-rewrite IC-LoRA path below it, which does route an image through enhance_i2v.
        assert ic_lora.prompt_template is not None
        system_prompt = build_template_fill_system_prompt(ic_lora)
        seed = self._random_seed()
        try:
            if req.provider == "api":
                resolved_model = resolve_gemini_model(self.state.app_settings.gemini_model)
                logger.info("Enhancing prompt via Gemini API (%s)", resolved_model)
                raw = self._gemini_pipeline.enhance_t2v(
                    req.prompt,
                    system_prompt=system_prompt,
                    seed=seed,
                    api_key=self.state.app_settings.gemini_api_key.strip(),
                    model=resolved_model,
                )
            else:
                logger.info("Enhancing prompt via local Gemma")
                assert gemma_root is not None
                pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
                raw = pipeline.enhance_t2v(req.prompt, system_prompt=system_prompt, seed=seed)
            values = parse_template_fill_response(raw, set(ic_lora.prompt_template.placeholders))
            return fill_prompt_template(ic_lora.prompt_template, values)
        except ValueError as e:
            raise HTTPError(500, f"PROMPT_TEMPLATE_FILL_FAILED: {e}") from e

    def _load_prompt_enhancer_pipeline(self, gemma_root: str) -> PromptEnhancerPipeline:
        self._pipelines.evict_gpu_pipeline_for_prompt_enhancement()
        device = get_device_type(self.config.device)
        return self._prompt_enhancer_pipeline_class.create(gemma_root, device)
