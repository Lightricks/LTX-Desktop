"""Pure helpers of the prompt enhancement handler: video roles and IC-LoRA rewrites."""

from __future__ import annotations

from collections.abc import Callable

from api_types import EnhancePromptRequest, IcLoraCatalogItem
from services.prompt_enhancement import (
    build_ic_lora_enhancement_system_prompt,
    build_template_fill_system_prompt,
    enforce_trigger_placements,
    fill_prompt_template,
    parse_template_fill_response,
)
from services.prompt_enhancement.i2v_frames import SceneFrame, VideoRole, reference_still
from services.prompt_enhancement.source_video import SourceVideo


def video_role(req: EnhancePromptRequest) -> VideoRole:
    """A list of panel images is its own role. An IC-LoRA transforms its video, so the stills
    are ground truth. Anything else (Extend, Retake) only gets scene context from them and the
    typed prompt leads."""
    if req.imagePaths:
        return "panels"
    return "driving" if req.icLoraId is not None else "scene"


def ic_lora_rewrite(ic_lora: IcLoraCatalogItem, run: Callable[[str], str]) -> str:
    """The catalog policy of an IC-LoRA rewrite, for every path that makes one.

    ``run`` sends one rewrite call with the given system prompt and returns the raw reply.
    A template entry gets the template fill. Any other entry gets a free rewrite with its
    trigger phrases enforced. ``ValueError`` means the reply did not fill the template.
    """
    template = ic_lora.prompt_template
    if template is not None:
        raw = run(build_template_fill_system_prompt(ic_lora))
        values = parse_template_fill_response(raw, set(template.placeholders))
        return fill_prompt_template(template, values)
    enhanced = run(build_ic_lora_enhancement_system_prompt(ic_lora))
    # Trigger enforcement on an empty rewrite would make a bare trigger word the prompt.
    if not enhanced.strip():
        return enhanced
    return enforce_trigger_placements(enhanced, [ic_lora])


def writes_blank_prompt(source_video: SourceVideo | None) -> bool:
    return source_video is not None and source_video.writes_blank_prompt


def reference_frame(ic_lora: IcLoraCatalogItem, path: str) -> SceneFrame:
    """A catalog entry pins the frame its look image conditions: the first frame of the clip
    (Restore), or a still before the clip (Layout to Render). An unpinned one uses frame 0."""
    pinned = ic_lora.reference_image_frame
    return reference_still(path, is_clip_frame=pinned is None or pinned >= 0)
