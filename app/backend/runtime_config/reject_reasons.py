"""Shared job-reject reason vocabulary.

One definition imported by the budget layer (decision types, HTTP mapping)
and the API layer (specs reason field), so a new reason flows through both
without a second Literal drifting. Imports nothing internal by design.
"""

from __future__ import annotations

from typing import Literal

# ``unmeasured_cell``: a priced load outside every lab-measured cell
# (``UnmeasuredBasis``). Produced only by admission: Cautious refuses it with the
# priced figures; Brave allows it with an unmeasured warning.
# ``one_pass_unavailable``: one pass was asked for a 1440p or 4K clip longer than one
# window, which those canvases always run in windows. A wall in every mode.
VideoJobRejectReason = Literal[
    "stream_over",
    "unsupported",
    "ram_shortfall",
    "mps_sequence_over",
    "unmeasured_cell",
    "one_pass_unavailable",
]

# How a Fast clip is cut. ``auto`` lets the budget pick (one pass when it fits, else
# windows); ``one_pass`` and ``parts`` are a user's explicit choice, or ``one_pass`` from
# the "Never split into parts" setting.
PassShape = Literal["auto", "one_pass", "parts"]

# Why a priced verdict is an extrapolation: a 1440p or 4K cell no run covers
# (measured_cells.py), a two-stage IC-LoRA window past the priced token range, or
# a Mac two-stage IC-LoRA shape outside the measured Layout to Render cell, or a Fast clip
# below 1440p that runs in more windows than any run, or a Fast single pass larger than the
# windows the laws were fitted on, or a new model not yet in the measured set.
UnmeasuredBasis = Literal[
    "high_res_cell",
    "two_stage_range",
    "mac_two_stage_cell",
    "windowed_range",
    "pass_tokens",
    "unmeasured_model",
]

# Hardware-budget mode: which measurement tiers block. "brave" (the default)
# turns an over-budget verdict into allow-with-warning and admits unmeasured
# loads with a warning; "cautious" refuses both. An unmeasured note does not
# cover a later memory miss. Hard refusals (unsupported, spatial, decode floor,
# MPS sequence) block in every mode. Stored in AppSettings, threaded onto
# LocalJobFilter, echoed on specs responses.
BudgetMode = Literal["brave", "cautious"]

# Warning tiers on a kept list cell. ``tight`` never confirms; ``over`` and
# ``unmeasured`` are admitted bets the mode allowed.
WarnedCellTier = Literal["tight", "over", "unmeasured"]

# Tiers a create-time annotation may carry. Tight fits never block anywhere,
# so they are never annotated.
JobRiskTier = Literal["over", "unmeasured"]

# Which figure bound a CUDA over-budget verdict. One spelling shared by the
# dropped cells, the risk annotation, and the budget helpers.
ShortfallKind = Literal["total", "free"]

# List-time IC-LoRA duration drops: the shared budget reasons plus the two
# IC-LoRA-only hard walls (spatial canvas cap, long Refine past the decode
# floor). Kept separate from VideoJobRejectReason because the shared vocabulary
# feeds the Fast/edit reject table, where these two can never occur. The budget
# reasons still pass through by name, so extending the shared literal fails
# typecheck at the IC-LoRA passthrough until this literal follows.
IcLoraDroppedReason = Literal[
    "stream_over",
    "unsupported",
    "ram_shortfall",
    "mps_sequence_over",
    "unmeasured_cell",
    "one_pass_unavailable",
    "spatial",
    "refine_decode_floor",
    "window_over_cap",
]

# Execute-time budget family: HTTP/capability codes meaning "the job stopped
# fitting between create and execute". The queue runner maps these to
# JOB_TOO_LARGE; the frontend passes them through verbatim. Literal strings,
# not imports: this module imports nothing internal by design (the budget
# constants live downstream). Pinned to shared/budget-miss-codes.json by
# test_budget_miss_codes_match_shared_contract — extend both together.
BUDGET_MISS_CODES: frozenset[str] = frozenset(
    {
        "IC_LORA_SOURCE_TOO_LARGE",
        "LOCAL_GENERATION_RAM_SHORTFALL",
        "LOCAL_GENERATION_UNSUPPORTED",
        "VIDEO_JOB_TOO_LARGE",
    }
)
