"""The generation stats projection. JSON paths for the dashboard live here.

fps is null unless the JSON value is a number, so a string like "24" cannot
pass for a real frame rate.
"""

from __future__ import annotations

from services.features.ic_lora_recipes import queued_ic_lora_recipes

_IC_LORA_IDS = ", ".join(
    f"'{recipe.recipe_id}'" for recipe in queued_ic_lora_recipes()
)
_IC_LORA_GROUP = (
    f"WHEN feature IN ({_IC_LORA_IDS}) THEN 'ic-lora'\n            "
    if _IC_LORA_IDS
    else ""
)

GENERATION_STATS_CTE = f"""
WITH generation_stats AS (
    SELECT
        id,
        feature,
        status,
        error_code,
        json_extract(spec, '$.params.model') AS model,
        CASE
            WHEN json_type(spec, '$.params.resolution') = 'text'
            THEN json_extract(spec, '$.params.resolution')
        END AS resolution_label,
        json_extract(spec, '$.params.aspectRatio') AS aspect_ratio,
        CASE
            WHEN json_type(spec, '$.params.fps') IN ('integer', 'real')
            THEN CAST(json_extract(spec, '$.params.fps') AS INTEGER)
        END AS fps,
        CASE
            {_IC_LORA_GROUP}WHEN feature IN (
                'text-to-video',
                'image-to-video',
                'audio-to-video',
                'retake',
                'extend'
            ) THEN feature
            WHEN COALESCE(json_array_length(spec, '$.params.loras'), 0) > 0 THEN 'lora'
            ELSE feature
        END AS feature_group,
        json_extract(spec, '$.params.loras[0].catalogId') AS lora_catalog_id,
        json_extract(spec, '$.params.loras[0].displayName') AS lora_name,
        CASE
            WHEN feature IN ('retake', 'extend') THEN NULL
            WHEN feature = 'audio-to-video'
                AND json_type(spec, '$.params.fps') IN ('integer', 'real')
                AND CAST(json_extract(spec, '$.params.fps') AS INTEGER) > 0
            THEN ROUND((
                CAST(json_extract(spec, '$.params.numFrames') AS REAL) - 1
            ) / CAST(json_extract(spec, '$.params.fps') AS INTEGER))
            ELSE CAST(json_extract(spec, '$.params.duration') AS REAL)
        END AS render_duration_s,
        CASE
            WHEN feature IN ('retake', 'extend')
            THEN CAST(json_extract(spec, '$.params.duration') AS REAL)
        END AS edited_span_s,
        created_at,
        started_at,
        finished_at,
        CASE
            WHEN started_at IS NOT NULL AND finished_at IS NOT NULL
            THEN finished_at - started_at
        END AS run_ms,
        deleted_at IS NOT NULL AS is_deleted
    FROM generations
    WHERE status IN ('succeeded', 'failed', 'cancelled')
)
"""


def generation_stats_sql(body: str) -> str:
    return GENERATION_STATS_CTE + body
