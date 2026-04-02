"""Text encoder patching and API embedding service."""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from services.http_client.http_client import HTTPClient
from state.app_state_types import TextEncodingResult

if TYPE_CHECKING:
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class LTXTextEncoder:
    """Stateless text encoding operations with idempotent monkey-patching."""

    def __init__(self, device: torch.device, http: HTTPClient, ltx_api_base_url: str) -> None:
        self.device = device
        self.http = http
        self.ltx_api_base_url = ltx_api_base_url
        self._prompt_encoder_patched = False
        self._cleanup_memory_patched = False

    def install_patches(self, state_getter: Callable[[], AppState]) -> None:
        self._install_prompt_encoder_init_patch()
        self._install_prompt_encoder_patch(state_getter)
        self._install_cleanup_memory_patch(state_getter)

    def _install_prompt_encoder_init_patch(self) -> None:
        """Patch PromptEncoder.__init__ to accept None gemma_root (API encoding mode).

        In API encoding mode, gemma_root is None since text encoding is done
        remotely.  The upstream PromptEncoder eagerly resolves file paths from
        gemma_root in __init__, which crashes.  This patch short-circuits init
        when gemma_root is falsy, creating a stub that the __call__ patch will
        intercept before any model loading.
        """
        try:
            from ltx_pipelines.utils.blocks import PromptEncoder

            original_init = PromptEncoder.__init__

            def patched_init(
                self_encoder: PromptEncoder,
                checkpoint_path: str,
                gemma_root: str,
                dtype: Any,
                device: Any,
                registry: Any = None,
            ) -> None:
                if not gemma_root:
                    self_encoder._dtype = dtype  # type: ignore[attr-defined]
                    self_encoder._device = device  # type: ignore[attr-defined]
                    self_encoder._text_encoder_builder = None  # type: ignore[attr-defined]
                    self_encoder._embeddings_processor_builder = None  # type: ignore[attr-defined]
                    return
                original_init(self_encoder, checkpoint_path, gemma_root, dtype, device, registry)

            PromptEncoder.__init__ = patched_init  # type: ignore[assignment]
            logger.info("Installed PromptEncoder.__init__ patch for None gemma_root")
        except Exception as exc:
            logger.warning("Failed to patch PromptEncoder.__init__: %s", exc, exc_info=True)

    def _install_prompt_encoder_patch(self, state_getter: Callable[[], AppState]) -> None:
        """Patch PromptEncoder.__call__ to use API embeddings when available."""
        if self._prompt_encoder_patched:
            return

        try:
            from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
            from ltx_pipelines.utils.blocks import PromptEncoder

            original_call = PromptEncoder.__call__

            def patched_call(
                self_encoder: PromptEncoder,
                prompts: list[str],
                **kwargs: Any,
            ) -> list[EmbeddingsProcessorOutput]:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is not None and te_state.api_embeddings is not None:
                    video_context = te_state.api_embeddings.video_context
                    audio_context = te_state.api_embeddings.audio_context
                    # Create a dummy attention mask matching the sequence length
                    seq_len = video_context.shape[1] if video_context.dim() > 1 else 1
                    dummy_mask = torch.ones(1, seq_len, device=video_context.device)
                    results: list[EmbeddingsProcessorOutput] = []
                    for i in range(len(prompts)):
                        if i == 0:
                            results.append(EmbeddingsProcessorOutput(
                                video_encoding=video_context,
                                audio_encoding=audio_context,
                                attention_mask=dummy_mask,
                            ))
                        else:
                            zero_video = torch.zeros_like(video_context)
                            zero_audio = torch.zeros_like(audio_context) if audio_context is not None else None
                            results.append(EmbeddingsProcessorOutput(
                                video_encoding=zero_video,
                                audio_encoding=zero_audio,
                                attention_mask=dummy_mask,
                            ))
                    return results

                return original_call(self_encoder, prompts, **kwargs)

            PromptEncoder.__call__ = patched_call  # type: ignore[assignment]

            self._prompt_encoder_patched = True
            logger.info("Installed PromptEncoder API embeddings patch")
        except Exception as exc:
            logger.warning("Failed to patch PromptEncoder: %s", exc, exc_info=True)

    def _install_cleanup_memory_patch(self, state_getter: Callable[[], AppState]) -> None:
        """Patch cleanup_memory to move cached text encoder to CPU before cleanup."""
        if self._cleanup_memory_patched:
            return

        try:
            from ltx_pipelines.utils import helpers as ltx_utils

            original_cleanup_memory = ltx_utils.cleanup_memory

            def patched_cleanup_memory() -> None:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is not None and te_state.cached_encoder is not None:
                    try:
                        te_state.cached_encoder.to(torch.device("cpu"))
                    except Exception:
                        logger.warning("Failed to move cached text encoder to CPU", exc_info=True)
                original_cleanup_memory()

            setattr(ltx_utils, "cleanup_memory", patched_cleanup_memory)

            for module_name in (
                "ltx_pipelines.utils.helpers",
                "ltx_pipelines.distilled",
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
                "ltx_pipelines.retake_pipeline",
            ):
                try:
                    module = __import__(module_name, fromlist=["cleanup_memory"])
                    if hasattr(module, "cleanup_memory"):
                        setattr(module, "cleanup_memory", patched_cleanup_memory)
                except Exception:
                    logger.warning("Failed to patch cleanup_memory for module %s", module_name, exc_info=True)

            self._cleanup_memory_patched = True
            logger.info("Installed cleanup_memory patch")
        except Exception as exc:
            logger.warning("Failed to patch cleanup_memory: %s", exc, exc_info=True)

    def get_model_id_from_checkpoint(self, checkpoint_path: str) -> str | None:
        try:
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata and "encrypted_wandb_properties" in metadata:
                    return metadata["encrypted_wandb_properties"]
        except Exception as exc:
            logger.warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
        return None

    def encode_via_api(self, prompt: str, api_key: str, checkpoint_path: str, enhance_prompt: bool) -> TextEncodingResult | None:
        model_id = self.get_model_id_from_checkpoint(checkpoint_path)
        if not model_id:
            return None

        try:
            start = time.time()
            response = self.http.post(
                f"{self.ltx_api_base_url}/v1/prompt-embedding",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json_payload={
                    "prompt": prompt,
                    "model_id": model_id,
                    "enhance_prompt": enhance_prompt,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.warning("LTX API error %s: %s", response.status_code, response.text)
                return None

            conditioning = pickle.loads(response.content)  # noqa: S301
            if not conditioning or len(conditioning) == 0:
                logger.warning("LTX API returned unexpected conditioning format")
                return None

            embeddings = conditioning[0][0]
            video_dim = 4096
            if embeddings.shape[-1] > video_dim:
                video_context = embeddings[..., :video_dim].contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = embeddings[..., video_dim:].contiguous().to(dtype=torch.bfloat16, device=self.device)
            else:
                video_context = embeddings.contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = None

            logger.info("Text encoded via API in %.1fs", time.time() - start)
            return TextEncodingResult(video_context=video_context, audio_context=audio_context)

        except Exception as exc:
            logger.warning("LTX API encoding failed: %s", exc, exc_info=True)
            return None


class DummyTextEncoder:
    pass
