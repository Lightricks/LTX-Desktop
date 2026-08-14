"""Text encoding cache and API embedding handler."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import LTXLocalModelId
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import (
    LTXLocalModelSpec,
    get_existing_cp_path,
    get_ltx_model_spec,
    get_model_cp_spec,
    is_cp_downloaded,
    resolve_active_ltx_model_id,
    resolve_downloaded_prompt_enhancer_cp,
)
from state.app_state_types import AppState, TextEncodingResult

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class TextHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)

    def _active_ltx_model_id(self) -> LTXLocalModelId | None:
        return resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )

    def active_ltx_model_spec(self) -> LTXLocalModelSpec | None:
        model_id = self._active_ltx_model_id()
        return None if model_id is None else get_ltx_model_spec(model_id)

    @with_state_lock
    def _get_cached_prompt(self, prompt: str, enhance_prompt: bool) -> TextEncodingResult | None:
        te = self.state.text_encoder
        if te is None:
            return None
        return te.prompt_cache.get((prompt.strip(), enhance_prompt))

    @with_state_lock
    def _cache_prompt(self, prompt: str, enhance_prompt: bool, result: TextEncodingResult) -> None:
        te = self.state.text_encoder
        if te is None:
            return

        max_size = self.state.app_settings.prompt_cache_size
        if max_size <= 0:
            return

        key = (prompt.strip(), enhance_prompt)
        if key in te.prompt_cache:
            del te.prompt_cache[key]
        elif len(te.prompt_cache) >= max_size:
            oldest = next(iter(te.prompt_cache))
            del te.prompt_cache[oldest]
        te.prompt_cache[key] = result

    @with_state_lock
    def _set_api_embeddings(self, result: TextEncodingResult | None) -> None:
        if self.state.text_encoder is not None:
            self.state.text_encoder.api_embeddings = result

    def clear_api_embeddings(self) -> None:
        self._set_api_embeddings(None)

    def should_use_local_encoding(self) -> bool:
        """Decide whether to use local text encoding based on availability.

        The user's ``use_local_text_encoder`` setting acts as a tiebreaker only
        when **both** the API key and the local encoder are available.  When only
        one option exists, that option is used regardless of the setting.
        """
        settings = self.state.app_settings.model_copy(deep=True)
        api_available = bool(settings.ltx_api_key)
        spec = self.active_ltx_model_spec()
        local_available = spec is not None and is_cp_downloaded(self.models_dir, spec.text_encoder_cp)
        api_usable = api_available and (spec is None or spec.supports_api_text_encoding)

        if api_usable and local_available:
            return settings.use_local_text_encoder  # setting is tiebreaker
        return local_available  # use whichever is available

    def prepare_text_encoding(self, prompt: str, enhance_prompt: bool) -> None:
        """Validate settings and prepare text embeddings for a generation run.

        Raises RuntimeError with a prefixed message if text encoding is
        misconfigured, the local encoder is missing, or API encoding fails
        with no local fallback.
        """
        settings = self.state.app_settings.model_copy(deep=True)
        api_available = bool(settings.ltx_api_key)
        spec = self.active_ltx_model_spec()
        local_available = spec is not None and is_cp_downloaded(self.models_dir, spec.text_encoder_cp)

        if spec is not None and not spec.supports_api_text_encoding and not local_available:
            raise RuntimeError(
                f"TEXT_ENCODING_NOT_CONFIGURED: LTX {spec.version_label} requires the local text "
                f"encoder ({get_model_cp_spec(spec.text_encoder_cp).description}); an LTX API key "
                "cannot encode prompts for this version. Download it in Settings."
            )

        if not api_available and not local_available:
            raise RuntimeError(
                "TEXT_ENCODING_NOT_CONFIGURED: To generate videos, you need to configure text encoding. "
                "Either enter an LTX API Key in Settings, or enable the Local Text Encoder."
            )

        use_local = self.should_use_local_encoding()
        gemma_root = self.resolve_gemma_root()
        embeddings = self._prepare_api_embeddings(prompt, enhance_prompt)

        if not use_local and embeddings is None and gemma_root is None:
            raise RuntimeError(
                "LTX API text encoding failed and local text encoder is not available. "
                "Please download the text encoder from Settings or check your API key."
            )

    def resolve_gemma_root(self) -> str | None:
        if not self.should_use_local_encoding():
            return None
        model_id = self._active_ltx_model_id()
        if model_id is None:
            return None
        return str(get_existing_cp_path(self.models_dir, get_ltx_model_spec(model_id).text_encoder_cp))

    def resolve_prompt_enhancer_root_if_downloaded(self) -> str | None:
        """Like `resolve_gemma_root`, but answers "is the checkpoint present" rather than
        "should generation prefer local text encoding" — `should_use_local_encoding()`'s
        API-key tiebreaker (which defaults to API when both are available) answers a different
        question than local Enhance availability, and gates on a setting the Enhance UI never
        shows. The frontend's own local-availability check (`getTextEncoderRecommendation`)
        already uses checkpoint presence alone; this mirrors that for the backend gate.

        Resolves the enhancer rather than the encoder, which differ on 2.5. Prefers
        Gemma 4 E2B when downloaded, then Gemma 3 from a 2.3 install.
        """
        spec = self.active_ltx_model_spec()
        if spec is None:
            return None
        cp_id = resolve_downloaded_prompt_enhancer_cp(self.models_dir, spec)
        if cp_id is None:
            return None
        return str(get_existing_cp_path(self.models_dir, cp_id))

    def _prepare_api_embeddings(self, prompt: str, enhance_prompt: bool) -> TextEncodingResult | None:
        if self.should_use_local_encoding():
            self.clear_api_embeddings()
            return None

        settings = self.state.app_settings.model_copy(deep=True)
        if not settings.ltx_api_key:
            self.clear_api_embeddings()
            return None

        # The LTX API rejects an empty prompt (returns None), but an empty prompt is valid for
        # some IC-LoRAs (e.g. outpainting fills from the scene). Local gemma encodes "" fine; to
        # match that in API mode — where there's no gemma fallback — encode a neutral placeholder
        # so we still get usable embeddings. Nothing to enhance for an empty prompt, so skip it.
        if not prompt.strip():
            prompt = " "
            enhance_prompt = False

        cached = self._get_cached_prompt(prompt, enhance_prompt)
        if cached is not None:
            self._set_api_embeddings(cached)
            return cached

        te = self.state.text_encoder
        if te is None:
            return None

        model_id = self._active_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        model_spec = get_ltx_model_spec(model_id)

        encoded = te.service.encode_via_api(
            prompt=prompt,
            api_key=settings.ltx_api_key,
            checkpoint_path=str(get_existing_cp_path(self.models_dir, model_spec.model_cp)),
            enhance_prompt=enhance_prompt,
            api_model_id=model_spec.api_text_encoder_model_id,
        )
        if encoded is not None:
            self._cache_prompt(prompt, enhance_prompt, encoded)
            self._set_api_embeddings(encoded)
        return encoded
