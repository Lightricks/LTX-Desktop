"""What the custom LoRA form takes from a Hugging Face model card."""

from __future__ import annotations

from api_types import CUSTOM_LORA_TRIGGER_MAX
from services.url_downloader.hf_model_card import parse_model_card

_CARD = """---
license: apache-2.0
instance_prompt: InkW4sh
widget:
  - text: a fox in InkW4sh style
  - text: a koi pond, InkW4sh
  - text: a mountain, InkW4sh
  - text: a fourth example that is over the limit
---

# Ink Wash

![preview](preview.png)

This LoRA gives a [watercolor](https://example.com) ink wash look
with soft edges.

## Usage
Use it with LTX.
"""


def test_the_card_gives_its_trigger_three_examples_and_first_paragraph() -> None:
    suggestions = parse_model_card(_CARD)

    assert suggestions.trigger == "InkW4sh"
    assert suggestions.examples == (
        "a fox in InkW4sh style",
        "a koi pond, InkW4sh",
        "a mountain, InkW4sh",
    )
    assert suggestions.description == "This LoRA gives a watercolor ink wash look with soft edges."


def test_a_card_with_a_broken_yaml_header_gives_a_description_and_no_error() -> None:
    for header in (
        "instance_prompt: [unclosed",
        "date: 2023-02-30",
        "nested: " + "[" * 2000 + "]" * 2000,
    ):
        suggestions = parse_model_card(f"---\n{header}\n---\n\nA soft look.\n")
        assert suggestions.trigger is None
        assert suggestions.examples == ()
        assert suggestions.description == "A soft look."


def test_a_trigger_longer_than_the_limit_is_dropped() -> None:
    sentence = "a " * CUSTOM_LORA_TRIGGER_MAX
    suggestions = parse_model_card(f"---\ninstance_prompt: {sentence}\n---\n")

    assert suggestions.trigger is None


def test_a_card_without_instance_prompt_gives_the_first_code_span_of_its_trigger_section() -> None:
    card = """---
library_name: diffusers
---

# Felt Cutout

This LoRA makes a **felt cutout** look.

## Recommended Trigger Words

**Main trigger word:**

- `F3ltCut0u7`

**Other helpful trigger words:**

- `felt cutout style`
"""

    suggestions = parse_model_card(card)

    assert suggestions.trigger == "F3ltCut0u7"
    assert suggestions.description == "This LoRA makes a felt cutout look."


def _card(body: str) -> str:
    return f"---\nlibrary_name: diffusers\n---\n\n{body}"


def test_trained_words_line_gives_the_first_item_and_drops_an_empty_or_over_long_value() -> None:
    first = parse_model_card(_card("# Gollum\n\n**Base model**: LTXV 2.3\n**Trained words**: g0llum, a pale creature\n"))
    phrase = parse_model_card(_card("# Pan\n\n**Trained words**: A 360-degree panoramic video\n"))
    empty = parse_model_card(_card("# Anime\n\n**Base model**: LTXV 2.3\n**Trained words**: \n"))
    too_long = parse_model_card(_card(f"# Face\n\n**Trained words**: {'word ' * 30}\n"))

    assert first.trigger == "g0llum"
    assert phrase.trigger == "A 360-degree panoramic video"
    assert empty.trigger is None
    assert too_long.trigger is None


def test_trigger_sentences_give_the_word_without_quotes_or_a_trailing_comma() -> None:
    cases = {
        'Trigger Word : "fx dance"': "fx dance",
        "**Trigger word:** `crtanim,` (first token of every prompt).": "crtanim",
        'Trigger word "SPROUT"': "SPROUT",
    }

    for line, expected in cases.items():
        assert parse_model_card(_card(f"# Title\n\n{line}\n")).trigger == expected, line


def test_trigger_heading_reads_a_code_block_or_quoted_item_and_the_first_value_wins() -> None:
    block = parse_model_card(_card("## Trigger Word\n\n```text\nsrx_commercial\n```\n\n## Recommended Strength\n"))
    quoted = parse_model_card(
        _card("## Trigger Word\n\n- 'C6b4rP8nk'\n\n## Other Example triggers\n\n- `cinematic cyberpunk aesthetic`\n")
    )
    intro = parse_model_card(
        _card("## Recommended Trigger Words\n\nUse one or more of the following:\n\n- `claymation style`\n- `stop motion clay`\n")
    )

    assert block.trigger == "srx_commercial"
    assert quoted.trigger == "C6b4rP8nk"
    assert intro.trigger == "claymation style"


def test_no_trigger_word_and_a_note_under_a_trigger_table_give_no_trigger() -> None:
    none = parse_model_card(_card("**No trigger word** — the LoRA is always active when loaded.\n"))
    table = parse_model_card(
        _card(
            "### Trigger words\n\n| Register | Female |\n| --- | --- |\n| Whispering | `(woman, whispering)` |\n\n"
            "> **Note:** Male whisper requires the extra word `quietly` to work.\n"
        )
    )

    assert none.trigger is None
    assert table.trigger is None


def test_example_prompts_and_a_description_without_metadata_or_generated_boilerplate() -> None:
    lines = parse_model_card(
        _card(
            '# Dance\n\nExample prompt : "fx dance figure made of honey"\n\n'
            "This LoRA makes a figure dance.\n"
        )
    )
    block = parse_model_card(
        _card("# Real\n\n## Example prompts\n\n```text\none\ntwo\nthree\nfour\n```\n")
    )
    muapi = parse_model_card(
        _card("# Style\n\n**Base model**: LTXV 2.3\n**Trained words**: \n\n## Usage (Python)\n\nGet your key from the site.\n")
    )
    generated = parse_model_card(
        _card("<!-- This model card has been generated automatically. -->\n\n# LoRA Finetune\n\n## Model description\n\nThis is a lora finetune of model: `Lightricks/LTX-Video`.\n")
    )

    assert lines.examples == ("fx dance figure made of honey",)
    assert lines.description == "This LoRA makes a figure dance."
    assert block.examples == ("one", "two", "three")
    assert muapi.description == ""
    assert generated.description == ""
