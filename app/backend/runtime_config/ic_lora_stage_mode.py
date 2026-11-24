"""IC-LoRA stage layout as one value, and the size of the pass that carries the LoRA.

Three layouts exist. ``skip_stage_2`` plus ``stage_2_ic_lora`` is a pair of booleans with
four combinations, and ``skip_stage_2=True`` with ``stage_2_ic_lora=True`` means nothing:
stage 2 does not run. The mode removes that combination. Derive it once from the resolved
settings, then pass the mode. The chunk rule and the job budget both read it.

No imports from other runtime modules, so both can import this one without a cycle.
"""

from __future__ import annotations

from enum import Enum


class IcLoraStageMode(Enum):
    SINGLE_STAGE = "single_stage"  # stage 2 is skipped; stage 1 carries the IC-LoRA
    STAGE_1_LORA = "stage_1_lora"  # two stages; the IC-LoRA runs on stage 1 only
    STAGE_2_LORA = "stage_2_lora"  # two stages; the IC-LoRA and reference also run on stage 2


def ic_lora_stage_mode(*, skip_stage_2: bool, stage_2_ic_lora: bool) -> IcLoraStageMode:
    """Stage layout of a job. ``stage_2_ic_lora`` is ignored when stage 2 is skipped."""
    if skip_stage_2:
        return IcLoraStageMode.SINGLE_STAGE
    if stage_2_ic_lora:
        return IcLoraStageMode.STAGE_2_LORA
    return IcLoraStageMode.STAGE_1_LORA


def heavy_pass_size(mode: IcLoraStageMode, stage1_width: int, stage1_height: int) -> tuple[int, int]:
    """Width and height of the largest pass. This is also the size of the output frame.

    Stage 1 for a single-stage job. The stage-2 canvas (twice stage 1) for both
    two-stage layouts, with or without the IC-LoRA on stage 2. The spatial cap and
    the job budget both read this size.
    """
    if mode is IcLoraStageMode.SINGLE_STAGE:
        return stage1_width, stage1_height
    return stage1_width * 2, stage1_height * 2
