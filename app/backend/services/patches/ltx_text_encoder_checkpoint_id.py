"""Desktop-only: LTXTextEncoder.get_model_id_from_checkpoint via header read.

Split from safetensors_metadata_fix so inference.apply_patches() does not import
services.text_encoder.

TODO: drop this desktop hook in a follow-up. It is the only patch that reaches
services.text_encoder.

Remove once safetensors supports read-only mmap (same as safetensors_metadata_fix).

Usage:
    import services.patches.ltx_text_encoder_checkpoint_id  # noqa: F401
"""

from __future__ import annotations

from inference.patches.safetensors_metadata_fix import read_safetensors_metadata
from services.text_encoder.ltx_text_encoder import LTXTextEncoder


def _patched_get_model_id_from_checkpoint(self: LTXTextEncoder, checkpoint_path: str) -> str | None:
    try:
        meta = read_safetensors_metadata(checkpoint_path) or {}
        if "encrypted_wandb_properties" in meta:
            return meta["encrypted_wandb_properties"]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
    return None


assert hasattr(LTXTextEncoder, "get_model_id_from_checkpoint"), (
    "LTXTextEncoder.get_model_id_from_checkpoint not found — patch needs updating."
)
LTXTextEncoder.get_model_id_from_checkpoint = _patched_get_model_id_from_checkpoint  # type: ignore[assignment]
