"""Selectors for `/v1/prompt-embedding`.

LTX 2.5 OS checkpoints identify the text encoder via `gemma_source_checkpoint`
metadata (`ltx_version` + `gemma_version`), not a usable `model_id`. The public
endpoint accepts exactly one of:

- `model_id` — legacy Comfy / LTX 2.3 (`encrypted_wandb_properties` on the checkpoint)
- `model` — `{ ltx_version, gemma_version }` for LTX 2.5+

Split 2.5 checkpoints omit `encrypted_wandb_properties`, so generation uses this
`model` override. Keep it in lockstep with the gateway's
`LTX_2_5_GEMMA_SOURCE_CHECKPOINT`.
"""

from typing import TypedDict


class LtxApiPromptEmbeddingModel(TypedDict):
    ltx_version: str
    gemma_version: str


# Same pair the LTX API video path uses for LTX-2.5 prompt-encode.
# Source: https://github.com/LightricksResearch/ltxv-api/pull/1604
LTX_2_5_API_PROMPT_EMBEDDING_MODEL: LtxApiPromptEmbeddingModel = {
    "ltx_version": "2.5.0",
    "gemma_version": "gemma4-12b-ltx-v1",
}
