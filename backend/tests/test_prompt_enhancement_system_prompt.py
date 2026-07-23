"""Unit tests for the pure prompt-enhancement helpers (no GPU, no Fake service needed)."""

from __future__ import annotations

import pytest
from api_types import (
    DownloadSpec,
    DownloadVariant,
    IcLoraCatalogItem,
    InputSpec,
    InstructionSection,
    LoraCatalogItem,
    PromptTemplatePlaceholder,
    PromptTemplateSpec,
)
from services.prompt_enhancement import (
    build_conditioning_system_prompt,
    build_ic_lora_enhancement_system_prompt,
    build_lora_enhancement_system_prompt,
    build_template_fill_system_prompt,
    enforce_trigger_placement,
    enforce_trigger_placements,
    fill_prompt_template,
    parse_template_fill_response,
    render_catalog_item_block,
)


def _dl(filename: str = "x.safetensors") -> DownloadSpec:
    return DownloadSpec(
        repo_id="org/x",
        variants=[DownloadVariant(id="default", label="Default", filename=filename, size_bytes=10)],
    )


def _lora(**overrides: object) -> LoraCatalogItem:
    defaults: dict[str, object] = dict(
        id="test-lora",
        name="Test Lora",
        description="d",
        download=_dl(),
        requires_hf_login=False,
    )
    defaults.update(overrides)
    return LoraCatalogItem(**defaults)


def _ic_lora(**overrides: object) -> IcLoraCatalogItem:
    defaults: dict[str, object] = dict(
        id="test-ic-lora",
        name="Test IC-LoRA",
        description="d",
        download=_dl(),
        requires_hf_login=False,
        input=InputSpec(kind="video"),
    )
    defaults.update(overrides)
    return IcLoraCatalogItem(**defaults)


# ============================================================
# render_catalog_item_block
# ============================================================


def test_block_includes_summary_and_prompting_excludes_tips_and_input():
    lora = _lora(
        instructions=[
            InstructionSection(kind="summary", title="What it does", body="Does the thing."),
            InstructionSection(kind="prompting", title="Prompt", body=["Say the thing.", "Example: the thing."]),
            InstructionSection(kind="tips", title="Tips", body="Strength 1.0."),
            InstructionSection(kind="input", title="Input", body="An image."),
        ]
    )
    block = render_catalog_item_block(lora)
    assert "Does the thing." in block
    assert "Say the thing." in block
    assert "Example: the thing." in block
    assert "Strength 1.0." not in block
    assert "An image." not in block


def test_block_includes_trigger_instruction_first_token():
    lora = _lora(trigger="TRIG", trigger_placement="first_token")
    block = render_catalog_item_block(lora)
    assert '"TRIG"' in block
    assert "first word" in block


def test_block_includes_trigger_instruction_anywhere():
    lora = _lora(trigger="TRIG", trigger_placement="anywhere")
    block = render_catalog_item_block(lora)
    assert '"TRIG"' in block
    assert "somewhere" in block


def test_block_omits_trigger_instruction_when_no_trigger():
    lora = _lora()
    block = render_catalog_item_block(lora)
    assert "Trigger word" not in block


def test_block_includes_enhancement_examples():
    lora = _lora(enhancement_examples=["ex one", "ex two"])
    block = render_catalog_item_block(lora)
    assert "ex one" in block
    assert "ex two" in block


def test_block_has_no_examples_section_when_none_given():
    lora = _lora()
    assert "Additional examples" not in render_catalog_item_block(lora)


# ============================================================
# build_lora_enhancement_system_prompt / build_ic_lora_enhancement_system_prompt
# ============================================================


def test_multi_lora_prompt_includes_every_lora_in_order():
    a = _lora(id="a", name="Alpha", trigger="ALPHA", trigger_placement="first_token")
    b = _lora(id="b", name="Beta", trigger="BETA", trigger_placement="anywhere")
    prompt = build_lora_enhancement_system_prompt([a, b])
    assert prompt.index("Alpha") < prompt.index("Beta")
    assert "ALPHA" in prompt and "BETA" in prompt
    assert "every listed trigger" in prompt


def test_lora_prompt_forbids_prefaces_and_markdown():
    a = _lora(id="a", name="Alpha")
    prompt = build_lora_enhancement_system_prompt([a])
    assert "No titles, headings, prefaces" in prompt
    assert "markdown" in prompt.lower()


def test_ic_lora_prompt_includes_the_item():
    ic = _ic_lora(name="Colorizer", trigger="COLORIZE", trigger_placement="anywhere")
    prompt = build_ic_lora_enhancement_system_prompt(ic)
    assert "Colorizer" in prompt
    assert "COLORIZE" in prompt


def test_ic_lora_prompt_forbids_prefaces_and_markdown():
    ic = _ic_lora(name="Colorizer")
    prompt = build_ic_lora_enhancement_system_prompt(ic)
    assert "No titles, headings, prefaces" in prompt
    assert "markdown" in prompt.lower()


# ============================================================
# fill_prompt_template
# ============================================================


def test_fill_prompt_template_substitutes_free_text_placeholders():
    spec = PromptTemplateSpec(
        template="Reference shows {reference}. COLORIZE {result}.",
        placeholders={"reference": PromptTemplatePlaceholder(), "result": PromptTemplatePlaceholder()},
    )
    result = fill_prompt_template(spec, {"reference": "a grey rabbit", "result": "a brown rabbit"})
    assert result == "Reference shows a grey rabbit. COLORIZE a brown rabbit."


def test_fill_prompt_template_degenerate_no_placeholders():
    spec = PromptTemplateSpec(template="upscale", placeholders={})
    assert fill_prompt_template(spec, {}) == "upscale"


def test_fill_prompt_template_rejects_missing_value():
    spec = PromptTemplateSpec(template="{a}", placeholders={"a": PromptTemplatePlaceholder()})
    with pytest.raises(ValueError):
        fill_prompt_template(spec, {})


def test_fill_prompt_template_rejects_extra_value():
    spec = PromptTemplateSpec(template="{a}", placeholders={"a": PromptTemplatePlaceholder()})
    with pytest.raises(ValueError):
        fill_prompt_template(spec, {"a": "x", "b": "y"})


def test_fill_prompt_template_enforces_enum_choices():
    spec = PromptTemplateSpec(
        template="crossview. new camera angle: {azimuth}.",
        placeholders={"azimuth": PromptTemplatePlaceholder(choices=["lower", "higher"])},
    )
    assert fill_prompt_template(spec, {"azimuth": "lower"}) == "crossview. new camera angle: lower."
    with pytest.raises(ValueError):
        fill_prompt_template(spec, {"azimuth": "sideways"})


# ============================================================
# enforce_trigger_placement
# ============================================================


def test_enforce_first_token_noop_when_already_leading():
    assert enforce_trigger_placement("TRIG do a thing", "TRIG", "first_token") == "TRIG do a thing"


def test_enforce_first_token_prepends_when_missing():
    assert enforce_trigger_placement("do a thing", "TRIG", "first_token") == "TRIG do a thing"


def test_enforce_first_token_prepends_when_present_but_not_leading():
    assert enforce_trigger_placement("do a TRIG thing", "TRIG", "first_token") == "TRIG do a TRIG thing"


def test_enforce_anywhere_noop_when_present():
    assert enforce_trigger_placement("do a TRIG thing", "TRIG", "anywhere") == "do a TRIG thing"


def test_enforce_anywhere_appends_when_missing():
    assert enforce_trigger_placement("do a thing", "TRIG", "anywhere") == "do a thing TRIG"


def test_enforce_trigger_placement_case_insensitive_match():
    assert enforce_trigger_placement("trig do a thing", "TRIG", "first_token") == "trig do a thing"


def test_enforce_trigger_placement_strips_trailing_punctuation_for_match():
    # trigger "crtanim," already present (sans the literal trailing comma) at the front.
    assert enforce_trigger_placement("crtanim a scene", "crtanim,", "first_token") == "crtanim a scene"


def test_enforce_trigger_placement_does_not_match_as_substring_of_a_longer_trigger():
    # Regression: a real catalog pair — "f4nt4sy" is a prefix of "f4nt4sy4n1m6". A plain
    # substring/startswith check would treat "f4nt4sy" as already present whenever only the
    # longer trigger is, silently skipping its own enforcement.
    text = "a f4nt4sy4n1m6 scene"
    assert enforce_trigger_placement(text, "f4nt4sy", "anywhere") == f"{text} f4nt4sy"
    assert enforce_trigger_placement(text, "f4nt4sy", "first_token") == f"f4nt4sy {text}"


# ============================================================
# enforce_trigger_placements (plural, multi-item composition)
# ============================================================


def test_enforce_placements_applies_every_item_with_a_trigger():
    a = _lora(trigger="ALPHA", trigger_placement="anywhere")
    b = _lora(trigger="BETA", trigger_placement="anywhere")
    result = enforce_trigger_placements("do a thing", [a, b])
    assert "ALPHA" in result and "BETA" in result


def test_enforce_placements_skips_items_without_a_trigger():
    a = _lora()
    result = enforce_trigger_placements("do a thing", [a])
    assert result == "do a thing"


def test_enforce_placements_only_one_first_token_leader():
    a = _lora(trigger="ALPHA", trigger_placement="first_token")
    b = _lora(trigger="BETA", trigger_placement="first_token")
    result = enforce_trigger_placements("do a thing", [a, b])
    # ALPHA claims the lead; BETA is downgraded to anywhere instead of clobbering it.
    assert result.startswith("ALPHA")
    assert "BETA" in result
    assert not result.startswith("BETA")


# ============================================================
# build_template_fill_system_prompt / parse_template_fill_response
# ============================================================


def test_template_fill_prompt_lists_placeholder_keys_and_choices():
    ic = _ic_lora(
        prompt_template=PromptTemplateSpec(
            template="crossview. new camera angle: {azimuth}.",
            placeholders={"azimuth": PromptTemplatePlaceholder(choices=["lower", "higher"])},
        )
    )
    prompt = build_template_fill_system_prompt(ic)
    assert '"azimuth"' in prompt
    assert "lower" in prompt and "higher" in prompt
    assert "JSON" in prompt


def test_template_fill_prompt_includes_item_block():
    ic = _ic_lora(
        name="Colorizer",
        prompt_template=PromptTemplateSpec(
            template="{reference} {result}",
            placeholders={"reference": PromptTemplatePlaceholder(), "result": PromptTemplatePlaceholder()},
        ),
    )
    assert "Colorizer" in build_template_fill_system_prompt(ic)


def test_parse_template_fill_response_plain_json():
    values = parse_template_fill_response('{"reference": "a rabbit", "result": "a brown rabbit"}', {"reference", "result"})
    assert values == {"reference": "a rabbit", "result": "a brown rabbit"}


def test_parse_template_fill_response_strips_markdown_fence():
    raw = '```json\n{"a": "x"}\n```'
    assert parse_template_fill_response(raw, {"a"}) == {"a": "x"}


def test_parse_template_fill_response_extracts_fence_after_leading_garbage():
    # Observed in the wild: the model prepends a dangling fragment before the fenced JSON.
    raw = (
        ', but the rabbit is now standing in a shallow pool of water reflecting the sky.\n'
        '```json\n{\n  "reference": "Rabbit on mossy rocks, dry scene",\n'
        '  "result": "Rabbit in shallow pool, sky reflection"\n}\n```'
    )
    values = parse_template_fill_response(raw, {"reference", "result"})
    assert values == {
        "reference": "Rabbit on mossy rocks, dry scene",
        "result": "Rabbit in shallow pool, sky reflection",
    }


def test_parse_template_fill_response_extracts_unfenced_json_from_prose():
    raw = 'Sure, here you go: {"a": "x"} — hope that helps!'
    assert parse_template_fill_response(raw, {"a"}) == {"a": "x"}


def test_parse_template_fill_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_template_fill_response("not json", {"a"})


def test_parse_template_fill_response_rejects_wrong_keys():
    with pytest.raises(ValueError):
        parse_template_fill_response('{"a": "x"}', {"b"})


def test_parse_template_fill_response_rejects_non_string_value():
    with pytest.raises(ValueError):
        parse_template_fill_response('{"a": 5}', {"a"})


# ============================================================
# build_conditioning_system_prompt (built-in canny/depth, no catalog entry)
# ============================================================


def test_depth_conditioning_prompt_forbids_reinventing_the_scene():
    prompt = build_conditioning_system_prompt("depth")
    assert "depth" in prompt.lower()
    assert "faithfully" in prompt.lower()
    assert "don't invent" in prompt.lower()


def test_canny_conditioning_prompt_forbids_reinventing_the_scene():
    prompt = build_conditioning_system_prompt("canny")
    assert "edge" in prompt.lower()
    assert "faithfully" in prompt.lower()


def test_conditioning_prompts_forbid_prefaces_and_markdown():
    for conditioning_type in ("canny", "depth"):
        prompt = build_conditioning_system_prompt(conditioning_type)
        assert "No titles, headings, prefaces" in prompt
