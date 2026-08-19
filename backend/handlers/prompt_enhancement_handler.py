"""Local, catalog-aware prompt enhancement handler."""

from __future__ import annotations

import logging
import random
import uuid
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import EnhancePromptRequest, EnhancePromptResponse, IcLoraCatalogItem, LoraCatalogItem
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from server_utils.media_validation import validate_image_file
from services.gemini_text_client import resolve_gemini_model
from services.interfaces import PromptEnhancerPipeline
from services.lora_catalog import LoraCatalogProvider
from services.prompt_enhancement import (
    build_audio_visual_caption_system_prompt,
    build_conditioning_system_prompt,
    build_ic_lora_enhancement_system_prompt,
    build_image_edit_system_prompt,
    build_image_generation_system_prompt,
    build_lora_enhancement_system_prompt,
    build_template_fill_system_prompt,
    enforce_trigger_placements,
    fill_prompt_template,
    parse_template_fill_response,
)
from services.prompt_enhancer_pipeline.gemini_prompt_enhancer_pipeline import GeminiPromptEnhancerPipeline
from services.services_utils import get_device_type
from state.app_state_types import AppState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

# Deliberately independent of StateHandlerBase._resolve_seed(): that helper honors the app's
# reproducibility seed lock (and a fixed constant in dev mode), which would make every enhance
# call — including a redo — produce the exact same output. Enhancement is a quick, exploratory
# action where a fresh draw each call is the whole point.
_MAX_ENHANCE_SEED = 2147483647


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
            elif not self.state.app_settings.gemini_api_key:
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
        return self._video_system_prompt(t2v=req.imagePath is None)

    def _video_system_prompt(self, *, t2v: bool) -> str | None:
        """The active model's own caption style, or None to keep each provider's default.

        Only the audio-visual generations (2.5) need this: their captions cover the soundscape,
        which neither the generic Gemini fallback nor a 2.3-era prompt asks for.
        """
        spec = self._text_handler.active_ltx_model_spec()
        if spec is None or not spec.wants_audio_visual_captions:
            return None
        return build_audio_visual_caption_system_prompt(t2v=t2v)

    def enhance_for_generation(self, prompt: str, *, image_path: str | None) -> str:
        """Rewrite ``prompt`` on the local enhancer for a generation that's already started.

        Only for the local text-encoding path: API encoding enhances server-side inside the
        same call, so it never gets here. Deliberately not `enhance()` — the caller already
        holds the generation slot, and there's no provider choice to make, only "is the
        enhancer on disk".

        Never raises: enhancement is a quality step, so a missing checkpoint or a failed
        rewrite degrades to the prompt as typed rather than failing the generation.
        """
        if not prompt.strip():
            return prompt
        gemma_root = self._text_handler.resolve_prompt_enhancer_root_if_downloaded()
        if gemma_root is None:
            logger.info("Skipping automatic enhancement: no local prompt enhancer downloaded")
            return prompt

        try:
            pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
            system_prompt = self._video_system_prompt(t2v=image_path is None)
            seed = self._random_seed()
            if image_path is not None:
                enhanced = pipeline.enhance_i2v(
                    prompt, image_path, system_prompt=system_prompt, seed=seed
                )
            else:
                enhanced = pipeline.enhance_t2v(prompt, system_prompt=system_prompt, seed=seed)
        except Exception:
            logger.warning("Automatic local enhancement failed; using the prompt as typed", exc_info=True)
            return prompt

        if not enhanced.strip():
            return prompt
        logger.info(
            "Enhanced prompt locally for generation (%d -> %d chars): %s",
            len(prompt),
            len(enhanced),
            enhanced,
        )
        return enhanced

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
        if req.imagePath is not None:
            validate_image_file(req.imagePath)

        seed = self._random_seed()
        if req.provider == "api":
            resolved_model = resolve_gemini_model(self.state.app_settings.gemini_model)
            logger.info("Enhancing prompt via Gemini API (%s)", resolved_model)
            api_key = self.state.app_settings.gemini_api_key
            if req.imagePath is not None:
                return self._gemini_pipeline.enhance_i2v(
                    req.prompt,
                    req.imagePath,
                    system_prompt=system_prompt,
                    seed=seed,
                    api_key=api_key,
                    model=resolved_model,
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
        if req.imagePath is not None:
            return pipeline.enhance_i2v(req.prompt, req.imagePath, system_prompt=system_prompt, seed=seed)
        return pipeline.enhance_t2v(req.prompt, system_prompt=system_prompt, seed=seed)

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
                    api_key=self.state.app_settings.gemini_api_key,
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
