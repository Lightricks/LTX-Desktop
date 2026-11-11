"""Pure, GPU-free helpers for the LoRA-aware local prompt enhancer.

Builds a Gemma system prompt from selected catalog LoRA(s)/IC-LoRA metadata, fills a catalog
item's fixed `prompt_template` deterministically, and enforces trigger placement on a free
rewrite. No GPU, no side effects — the handler decides which path (free rewrite vs template
fill) to take based on whether the resolved item has a `prompt_template`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal, Protocol, cast

from api_types import IcLoraCatalogItem, InstructionSection, LoraCatalogItem, PromptTemplateSpec, TriggerPlacement
from services.prompt_enhancement.i2v_frames import VideoRole

_INCLUDED_KINDS = ("summary", "prompting")


class LoraPromptGuidance(Protocol):
    """The prompt-enhancement fields of a catalog LoRA or a saved custom LoRA."""

    @property
    def name(self) -> str: ...

    @property
    def instructions(self) -> Sequence[InstructionSection]: ...

    @property
    def trigger(self) -> str | None: ...

    @property
    def trigger_placement(self) -> TriggerPlacement | None: ...

    @property
    def enhancement_examples(self) -> Sequence[str]: ...


def _instruction_text(instructions: Sequence[InstructionSection], kind: str) -> list[str]:
    texts: list[str] = []
    for instr in instructions:
        if instr.kind != kind:
            continue
        texts.extend(instr.body if isinstance(instr.body, list) else [instr.body])
    return texts


def _trigger_instruction(item: LoraPromptGuidance) -> str | None:
    if item.trigger is None or item.trigger_placement is None:
        return None
    if item.trigger_placement == "first_token":
        return f'Trigger word: "{item.trigger}" — must be the first word of the rewritten prompt.'
    return f'Trigger word: "{item.trigger}" — must appear somewhere in the rewritten prompt.'


# The examples are target prompts, not a template. Without this the model copies their wording.
_EXAMPLES_LABEL = (
    "Example prompts for this LoRA. Match their style and level of detail. "
    "Do not copy their wording or their subject:"
)


def render_catalog_item_block(item: LoraPromptGuidance, *, include_trigger: bool = True) -> str:
    """Render one LoRA/IC-LoRA's catalog metadata as a block of text for Gemma to read.

    Only `summary`/`prompting`-kind instructions are included — `tips`/`input` are human-facing
    (strength recommendations, checkpoint choice, input-media notes) and would only add noise to
    the rewrite. `enhancement_examples` (never shown in the LoRA info UI) are appended as extra
    few-shot grounding. Without the trigger line for callers that add the trigger themselves.
    """
    lines = [f"### {item.name}"]
    for kind in _INCLUDED_KINDS:
        lines.extend(_instruction_text(item.instructions, kind))
    trigger_instruction = _trigger_instruction(item) if include_trigger else None
    if trigger_instruction:
        lines.append(trigger_instruction)
    if item.enhancement_examples:
        lines.append(_EXAMPLES_LABEL)
        lines.extend(f"- {example}" for example in item.enhancement_examples)
    return "\n".join(lines)


# Gemma's own default system prompt (used for the no-selection generic fallback) has a whole
# "Output Format (Strict)" section forbidding prefaces/markdown/titles. Our custom preambles
# replace that default entirely, so without this they lose that constraint — a real rewrite
# came back prefaced with "Here's the rewritten prompt..." and wrapped in markdown bold because
# nothing told it not to.
_OUTPUT_FORMAT_INSTRUCTION = (
    "Respond with ONLY the rewritten prompt, as a single continuous paragraph of natural "
    "language. No titles, headings, prefaces, explanations, quotes, or markdown formatting "
    "(no **bold**, no bullet points) — just the prompt text itself."
)


def build_default_free_rewrite_system_prompt() -> str:
    """Fallback system prompt for the generic (no LoRA/IC-LoRA/conditioning selected) free-rewrite
    path, for callers with no domain-tuned default of their own to fall back on. The local Gemma
    pipeline has its own baked-in default (``text_encoder.default_gemma_t2v_system_prompt`` /
    ``..._i2v_...``) that already includes output-format discipline — this exists for providers
    that don't, e.g. the Gemini API path, which otherwise gets zero system instruction and
    responds with a chatty, markdown-formatted essay instead of a rewritten prompt.
    """
    return (
        "You are a prompt engineer for a text-to-video generation model. Rewrite the user's "
        "prompt into a single, vivid, concrete visual description — subject, action, setting, "
        "lighting, and camera framing — preserving the user's original subject and intent "
        "without inventing an unrelated scene. " + _OUTPUT_FORMAT_INSTRUCTION
    )


def _clip_timing_line(duration: int | float | None, fps: int | float | None) -> str | None:
    # ponytail: both-or-neither; callers send duration+fps together
    if duration is not None and duration > 0 and fps is not None and fps > 0:
        return f"Clip: {duration:g} seconds at {fps} fps."
    return None


def build_i2v_user_prompt_text(
    prompt: str,
    *,
    has_last: bool,
    keyframe_count: int = 0,
    duration: int | float | None = None,
    fps: int | float | None = None,
) -> str:
    """User-turn text for an image-conditioned enhance call.

    First-only with a typed prompt keeps the Gemma wording ``User Raw Input Prompt: …``.
    Empty prompt + image(s) asks the VLM to caption from the frame(s) instead.
    Multi-keyframe treats any typed text as extra instructions on top of the stills,
    and prefixes clip duration/fps when the caller knows them.
    """
    stripped = prompt.strip()
    interpolating = has_last or keyframe_count > 1
    if keyframe_count > 0:
        if stripped:
            body = f"Extra user instructions: {stripped}"
        else:
            body = (
                "No extra user instructions. Infer the connecting motion from the "
                "keyframes in order."
            )
        clip = _clip_timing_line(duration, fps)
        return f"{clip}\n{body}" if clip is not None else body
    if not stripped:
        if interpolating:
            return (
                "No user prompt. Write a video generation prompt that interpolates "
                "from the first frame to the last."
            )
        return "No user prompt. Write a generation prompt from the attached frame."
    if interpolating:
        return f"User Raw Input Prompt: {stripped}."
    return f"User Raw Input Prompt: {prompt}."


def _no_guess_rule(subject: str) -> str:
    """The rule against detail an image does not show. ``subject`` is "a still" or "a panel"."""
    return (
        f"Do not guess wardrobe, accessories, or colors that {subject} does not show "
        "— stay vague if they are unclear."
    )


_VIDEO_FRAMES_COMMON_RULES = (
    "The user turn contains stills sampled in time order from a source video, each labeled "
    "with its position in the clip. Do not describe the stills one by one: write about the "
    "clip as a whole. If earlier instructions describe a single reference image or an exact "
    "first frame, ignore that framing. Keep the output format required above exactly."
)

_VIDEO_ROLE_RULES: dict[VideoRole, str] = {
    "driving": (
        "The video is the input this IC-LoRA transforms. Treat the stills as ground truth "
        "for the scene: its subjects, setting, action, and camera. Use the user's text as the "
        "instruction for what the IC-LoRA must do to that scene. Do not invent subjects, "
        "places, or events the stills do not show. If the user gave no text, derive the "
        "instruction from the IC-LoRA description above and what the stills show. "
        "Use that description to know what the IC-LoRA does, but do not reuse its wording: "
        "write the shot, not what the IC-LoRA does. "
        "A still labeled as a look reference image shows the look the result must have: the "
        "colors, the materials, the light, and the style. Write the prompt for the clip's "
        "subjects, setting, action, and camera, rendered in that look. Its label says whether "
        "it is a frame of the clip. If it is not, take no subjects, places, or events from it "
        "and use the other stills for them. If it is the finished first frame of the clip, it "
        "also shows how the clip's opening subjects and setting look when finished. "
    ),
    "panels": (
        "The user turn contains the constructed reference sheet image of this IC-LoRA, laying "
        "out each character, prop, or location in its own panel across rows. Treat the sheet as "
        "ground truth. Write the full prompt in two labeled parts, like the example prompts "
        "above. Start with Reference sheet: and describe every panel in detail with its row and "
        "position (for example, Top Row Left (Character): ..., Bottom Row (Setting): ...): its "
        "role, and its look, clothing, colors, and materials. End with Generated video: and "
        "write one long, detailed scene: the shot and the camera, who is in it and what each "
        "one does, any spoken words in quotes, the light, and the background. Use the panels "
        "for every subject in the scene. Describe only what the sheet shows. One panel shows "
        "one subject, even from several sides or poses: write it once and do not count views "
        "as separate subjects. "
        + _no_guess_rule("a panel")
        + " Do not add a subject, prop, or place that has no panel. "
        "For the Generated video part, build on the user's text and keep its action and "
        "intent. If the user gave no text, write one simple action that uses the panels. "
        "Ignore any advice above to keep the prompt short. Keep the output format required "
        "above: one paragraph."
    ),
    "scene": (
        "The video is an existing clip that will be edited or continued. Use the stills only "
        "for general context: the setting, the subjects, the lighting, and the visual style. "
        "The user's text is the instruction and has the most weight. Follow it, and do not "
        "add events, subjects, or camera moves it does not ask for. Do not retell what the "
        "stills already show. Add detail only where the user's text or the stills support it. "
        "If the user gave no text, write a prompt that fits the scene shown. "
        "Some stills are labeled as before or after a part of the clip that is being "
        "replaced. That part is not shown. The user's text describes what happens in the "
        "replaced part, so write it to fit the stills around it. "
    ),
}


# The `panels` rewrite is a two-part prompt of 300 to 700 words. The default cap of 512
# tokens on both providers ends it inside the `Generated video:` half.
_PANELS_MAX_OUTPUT_TOKENS = 1536


def video_role_max_output_tokens(role: VideoRole) -> int | None:
    """The output cap a role needs above the provider default, or None to keep the default."""
    return _PANELS_MAX_OUTPUT_TOKENS if role == "panels" else None


def with_video_role_rules(system_prompt: str, role: VideoRole) -> str:
    """``system_prompt`` plus the rules for how a source video's stills steer the rewrite.

    Applied by each provider after it resolves its own default, so the rules ride on the
    IC-LoRA prompt, the model's caption style, or the generic fallback alike.
    """
    # Panels are not stills of a clip, so the rules about time order and the clip do not apply.
    common = "" if role == "panels" else _VIDEO_FRAMES_COMMON_RULES
    return "\n\n".join([system_prompt, _VIDEO_ROLE_RULES[role] + common])


_LOOK_REFERENCE_REQUEST = (
    "Write the shot itself, not the task. Give the look of the look reference image in "
    "concrete plain words: the light and the time of day, the season, the colors, and the "
    "surfaces and materials. Use short phrases, and end with a few words on the finish. "
    "Never mention the reference image or the stills."
)


def build_video_user_prompt_text(
    prompt: str, *, role: VideoRole, has_look_reference: bool = False
) -> str:
    """User-turn text of a video-aware enhance call. A typed prompt is marked as the instruction.

    A look reference image is asked for here as well as in the system prompt. The model
    weighs the user turn more, and a catalog entry that asks for one short sentence would
    otherwise leave the look out.
    """
    stripped = prompt.strip()
    look = f" {_LOOK_REFERENCE_REQUEST}" if has_look_reference and role == "driving" else ""
    if stripped:
        if role in ("driving", "panels"):
            return f"User instruction for this video: {stripped}{look}"
        return f"User prompt (primary, follow it): {stripped}"
    if role == "panels":
        return "No user prompt. Describe the panels from the images, and write one simple action that uses them."
    if role == "driving":
        return f"No user prompt. Write the prompt from the stills and the IC-LoRA description.{look}"
    return "No user prompt. Write a prompt that fits the scene shown in the stills."


# Multi-keyframe enhance is stills-as-ground-truth, not a t2v rewrite. Distilled from the
# Atlas shot-control enhancer so interpolations stay identity-consistent and physically
# continuous, while keeping this app's single-paragraph output contract (no JSON, no MM:SS.ff).
_KEYFRAME_RULES = (
    "You are writing a prompt for a multi-keyframe video model. The user turn contains "
    "stills labeled by frame index, clock time, and lock strength, plus clip duration/fps "
    "and optional extra user instructions.\n"
    "The stills are visual ground truth. Each still's strength (0 to 1) is how tightly the "
    "video must match that still at its time: 1 is a full lock on the visible state; lower "
    "values allow more interpolation around that pose without changing the still's identity. "
    "For each still, first describe that still's visible "
    "state (pose, expression, staging, clothing as shown), then the connecting action that "
    "gets from it to the next still. Do not invent a blend or compromise pose that is in none "
    "of the stills. " + _no_guess_rule("a still") + " "
    "At each labeled time the video must match that still: "
    "same subject identity, setting, lighting, and composition. A person who shares core "
    "appearance across stills is ONE person who moved — never a second copy.\n"
    "Write a single continuous take that interpolates fluidly between the stills in time "
    "order. Describe the physical action that connects each pose or position (walk, turn, "
    "reach, sit) — never teleport. Keep constants (identity, persistent props, environment) "
    "in consistent language. Invent in-between motion only when the stills imply it; do not "
    "add characters, props, or locations that appear in no still and no extra instructions.\n"
    "Place action in time with phrases like \"at the start\", \"by two seconds\", and "
    "\"at the end\" when they help; do not emit JSON, MM:SS.ff timestamps, or a second "
    "paragraph.\n"
    "Extra user instructions, if present, supply narrative intent, pacing, and action "
    "between frames. They must not override what a still shows at its time. If they conflict "
    "with a still, keep the still's visible state and use the instructions only to shape "
    "energy and the in-between motion. If there are no extra instructions, infer that "
    "connecting motion from the stills alone.\n"
    "If earlier instructions describe a single reference image or exact first frame, ignore "
    "that framing: the user turn has labeled stills, each of which is visual ground truth at "
    "its time. Start immediately with the opening visual. Use progressive verbs throughout. "
    "No hard cuts."
)


def build_keyframe_enhancement_system_prompt(*, audio_visual: bool = False) -> str:
    """System prompt for multi-keyframe enhance (local Gemma and Gemini).

    2.3 is the keyframe rules alone. 2.5 keeps the model's native audio-visual caption
    style and appends those rules — it does not replace the native caption instructions
    with a shorter soundscape addendum.
    """
    rules = _KEYFRAME_RULES + " " + _OUTPUT_FORMAT_INSTRUCTION
    if not audio_visual:
        return rules
    return build_audio_visual_caption_system_prompt(t2v=False) + "\n\n" + rules


def build_audio_visual_caption_system_prompt(*, t2v: bool) -> str:
    """The captioning instructions LTX 2.5 itself ships with, for any enhancer provider.

    Read from ltx-core rather than restated here so it tracks the model: these describe the exact
    caption style the checkpoint was trained on (single 150-220 word paragraph, shot type + camera
    motion + viewpoint, and a full soundscape including quoted dialogue). The local Gemma enhancer
    picks this up on its own from the encoder's model type; providers with no model-side default —
    Gemini — would otherwise get the visual-only generic fallback and drop the audio half.
    """
    from ltx_core.text_encoders.gemma.encoders.base_encoder import (
        default_gemma4_i2v_system_prompt,
        default_gemma4_t2v_system_prompt,
    )

    return default_gemma4_t2v_system_prompt() if t2v else default_gemma4_i2v_system_prompt()


def build_image_generation_system_prompt() -> str:
    """System prompt for the image-generation (text-to-image) free-rewrite path, Z-Image-Turbo.

    Grounded in Tongyi-MAI/Z-Image-Turbo's model card and the FAL ``fal-ai/z-image/turbo``
    endpoint docs: the model responds well to vivid, concrete visual detail (subject, texture,
    lighting, composition) over abstract description, and has strong bilingual (English/Chinese)
    text rendering — in-image text should be kept exact rather than paraphrased.
    """
    return (
        "You are a prompt engineer for Z-Image-Turbo, a fast text-to-image diffusion model. "
        "Rewrite the user's prompt into a single, vivid, concrete visual description — subject, "
        "pose, materials/texture, lighting, and composition — specific rather than abstract. "
        "If the user wants text rendered in the image, keep it exact and put it in quotes; "
        "Z-Image-Turbo renders English and Chinese text accurately when given it verbatim. "
        "Preserve the user's original subject and intent; don't invent an unrelated scene. "
        + _OUTPUT_FORMAT_INSTRUCTION
    )


def build_image_edit_system_prompt() -> str:
    """System prompt for the image-editing (img2img) free-rewrite path, Z-Image-Turbo.

    Per FAL's ``fal-ai/z-image/turbo/image-to-image`` docs, editing is prompt + reference image +
    a strength dial (higher = more dramatic reimagining, lower = more of the source preserved) —
    describe the desired result, not an imperative "change X to Y" instruction.
    """
    return (
        "You are a prompt engineer for Z-Image-Turbo's image-to-image editing pipeline. The user "
        "has provided a reference image (shown to you below) and a rewrite strength that controls "
        "how much of it is preserved. Rewrite the user's instruction into a single, vivid "
        "description of the DESIRED RESULT image — describe the outcome, not a command like "
        "'change X to Y'. Reference concrete details already visible in the source image "
        "(subject, pose, setting) so the parts the user didn't ask to change stay consistent, and "
        "describe clearly what should be different. If the user wants text rendered in the image, "
        "keep it exact and in quotes. " + _OUTPUT_FORMAT_INSTRUCTION
    )


_CONDITIONING_DESCRIPTIONS: dict[Literal["canny", "depth"], str] = {
    "depth": "depth-map-conditioned control transform — the model already sees the exact geometry",
    "canny": "edge-map-conditioned control transform — the model already sees the exact edges/geometry",
}


def build_conditioning_system_prompt(conditioning_type: Literal["canny", "depth"]) -> str:
    """System prompt for the built-in (non-catalog) canny/depth conditioning modes of the
    "bring your own IC-LoRA" custom flow. These have no catalog entry to draw a system prompt
    from, but still need the same "describe the reference scene faithfully, don't invent a
    different one" discipline a catalog IC-LoRA's template-fill path gets — Gemma's default
    free-rewrite system prompt actively encourages inventing new characters/settings, which is
    wrong here: the control signal is conditioned on the real reference video's content.
    """
    description = _CONDITIONING_DESCRIPTIONS[conditioning_type]
    return (
        f"This is a {description} and motion of the reference video. Describe the same real "
        "scene faithfully: don't invent a different visual style, characters, or content not "
        "implied by the reference. You may adjust textures, materials, lighting, and anything "
        "the reference structure doesn't dictate. " + _OUTPUT_FORMAT_INSTRUCTION
    )


def build_lora_enhancement_system_prompt(loras: Sequence[LoraPromptGuidance]) -> str:
    """System prompt for the free-rewrite path with one or more selected plain LoRAs.

    Callers must not pass an item whose `prompt_template` is set — that item goes through the
    template-fill path (`fill_prompt_template`) instead of a free rewrite.
    """
    preamble = (
        "Rewrite the user's prompt so it makes full use of every LoRA selected below. "
        "Include every listed trigger word exactly as written, respecting its placement rule. "
        "Blend the LoRAs' styles/subjects into one coherent prompt — don't just concatenate "
        "them. Keep the user's original subject and intent; add only what the LoRAs need. "
        + _OUTPUT_FORMAT_INSTRUCTION
    )
    blocks = [render_catalog_item_block(lora) for lora in loras]
    return "\n\n".join([preamble, *blocks])


def build_ic_lora_enhancement_system_prompt(ic_lora: IcLoraCatalogItem) -> str:
    """System prompt for the free-rewrite path with a selected IC-LoRA.

    Callers must not pass an item whose `prompt_template` is set — see
    `build_lora_enhancement_system_prompt`.
    """
    preamble = (
        "Rewrite the user's prompt so it makes full use of the IC-LoRA selected below. "
        "Include its trigger word exactly as written, respecting its placement rule. "
        "Keep the user's original subject and intent; add only what the IC-LoRA needs. "
        + _OUTPUT_FORMAT_INSTRUCTION
    )
    return "\n\n".join([preamble, render_catalog_item_block(ic_lora)])


def fill_prompt_template(spec: PromptTemplateSpec, values: dict[str, str]) -> str:
    """Deterministically stitch `values` into `spec.template` — Gemma never touches the fixed
    scaffold text, only supplies the placeholder content, so the required shape can't drift.
    """
    if set(values) != set(spec.placeholders):
        raise ValueError(
            f"template values {sorted(values)} must exactly match placeholders {sorted(spec.placeholders)}"
        )
    for name, placeholder in spec.placeholders.items():
        if placeholder.choices is not None and values[name] not in placeholder.choices:
            raise ValueError(f"'{name}' value {values[name]!r} must be one of {placeholder.choices}")
    # The template owns the full stop after a placeholder. A value that ends with one too would
    # make "..", so the trailing stops of that value go.
    stitched = {
        name: value.rstrip().rstrip(".") if f"{{{name}}}." in spec.template else value
        for name, value in values.items()
    }
    return spec.template.format(**stitched)


def enforce_trigger_placement(text: str, trigger: str, placement: TriggerPlacement) -> str:
    """Guarantee `trigger` is present (and, for `first_token`, leading) in `text`.

    Post-processes a free rewrite rather than trusting the LLM to reliably obey a positional
    instruction — verified missing/misplaced triggers are fixed deterministically instead.

    Matches on a word boundary, not a plain substring/prefix — the catalog has triggers where
    one is a prefix of another (e.g. "f4nt4sy" / "f4nt4sy4n1m6"), and a substring check would
    treat the shorter trigger as "already present" whenever only the longer one actually is,
    silently skipping its enforcement.
    """
    stripped_trigger = trigger.rstrip(",.").lower()
    escaped_trigger = re.escape(stripped_trigger)
    text = text.strip()
    if placement == "first_token":
        if re.match(rf"^{escaped_trigger}\b", text.lower()):
            return text
        return f"{trigger} {text}"
    if re.search(rf"\b{escaped_trigger}\b", text.lower()):
        return text
    return f"{text} {trigger}"


def enforce_trigger_placements(text: str, items: Sequence[LoraPromptGuidance]) -> str:
    """Apply `enforce_trigger_placement` for every selected item that has one.

    Only one trigger can lead a prompt — if more than one selected item wants
    `first_token`, every one after the first is downgraded to `anywhere` so it doesn't
    clobber the one that already claimed the lead position.
    """
    first_token_claimed = False
    for item in items:
        if item.trigger is None or item.trigger_placement is None:
            continue
        placement = item.trigger_placement
        if placement == "first_token":
            if first_token_claimed:
                placement = "anywhere"
            else:
                first_token_claimed = True
        text = enforce_trigger_placement(text, item.trigger, placement)
    return text


def build_template_fill_system_prompt(item: LoraCatalogItem) -> str:
    """System prompt for the template-fill path: Gemma supplies only the placeholder content
    for `item.prompt_template`, never the fixed scaffold text.
    """
    assert item.prompt_template is not None, "build_template_fill_system_prompt requires prompt_template"
    placeholder_names = sorted(item.prompt_template.placeholders)
    lines = [
        "The user has a raw idea for a video. This LoRA/IC-LoRA requires an exact fixed prompt "
        "structure, given below for context — you never write that fixed wording yourself, you "
        "only supply the missing pieces.",
        f"Respond with ONLY a JSON object with exactly these keys: {placeholder_names}. "
        "No markdown, no code fences, no commentary — valid JSON only.",
    ]
    for name in placeholder_names:
        placeholder = item.prompt_template.placeholders[name]
        if placeholder.choices is not None:
            lines.append(f'- "{name}" must be exactly one of: {placeholder.choices}.')
        else:
            lines.append(f'- "{name}": a short descriptive phrase, grounded in the user\'s idea.')
    lines.append(render_catalog_item_block(item))
    return "\n".join(lines)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


_NEGATIVE_PROMPT_EXCLUSION_RULES = (
    "The user turn includes a list of excluded elements (negative prompt). "
    "Do NOT include, mention, describe, or imply any of those excluded elements, styles, or actions "
    "in your rewritten prompt, even if they appear in a reference image or video frame. "
    "Describe the scene without them, and do not write negative phrasing (like 'no X' or 'without X') — "
    "only describe what IS present in the desired shot."
)


def with_negative_prompt_rules(system_prompt: str, negative_prompt: str) -> str:
    if not negative_prompt.strip():
        return system_prompt
    return f"{system_prompt}\n\n{_NEGATIVE_PROMPT_EXCLUSION_RULES}"


def with_user_negative_prompt(user_text: str, negative_prompt: str) -> str:
    neg = negative_prompt.strip()
    if not neg:
        return user_text
    return f"{user_text}\nExcluded elements (do not include or mention in the prompt): {neg}"


def parse_template_fill_response(raw: str, placeholder_names: set[str]) -> dict[str, str]:
    """Parse Gemma's template-fill response, defensively extracting the JSON object from
    surrounding text (a leading/trailing fragment, or a markdown code fence) despite being told
    to respond with only JSON — the model doesn't always comply.
    """
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)
    else:
        # No fence — fall back to the widest {...} span, in case the object is embedded in
        # prose without one.
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            text = text[first : last + 1]

    try:
        parsed_raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"template-fill response was not valid JSON: {raw!r}") from e
    if not isinstance(parsed_raw, dict):
        raise ValueError(f"template-fill response must be a JSON object: {raw!r}")
    # JSON object keys are always strings — safe to narrow after the isinstance(dict) check.
    parsed = cast("dict[str, object]", parsed_raw)
    if set(parsed) != placeholder_names:
        raise ValueError(
            f"template-fill response keys {sorted(parsed)} must exactly match {sorted(placeholder_names)}"
        )
    values: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(value, str):
            raise ValueError(f"template-fill value for {key!r} must be a string, got {type(value).__name__}")
        values[key] = value
    return values
