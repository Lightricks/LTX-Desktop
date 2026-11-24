"""Suggestions for the custom LoRA form, read from a Hugging Face model card.

A model card is a README with an optional YAML header. Three things in it help prompt
enhancement: the trigger word, a short description, and example prompts. The reader never
fails: a card that has none of them gives empty suggestions, and the user fills the form.

- Trigger: a YAML key (``instance_prompt`` / ``trigger_word``), an explicit ``Trigger word:``
  or ``Trained words:`` label, or the first value under a ``# Trigger Word`` heading.
- Examples: the YAML ``widget`` texts, ``Example prompt: "..."`` lines, or the lines under an
  ``# Example Prompts`` heading.
- Description: the first plain paragraph before the first section heading, or under a heading
  such as ``Model description``. Generated boilerplate gives no description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

import yaml

from api_types import (
    CUSTOM_LORA_DESCRIPTION_MAX,
    CUSTOM_LORA_EXAMPLE_MAX,
    CUSTOM_LORA_EXAMPLES_MAX,
    CUSTOM_LORA_TRIGGER_MAX,
)

# TODO(security): CWE-400 (resource exhaustion). The card is untrusted text. ``yaml.safe_load``
# builds plain data only (no Python objects, CWE-502). The header is cut to this size so a
# crafted header cannot make the parser run long. Every pattern below is linear: none nests a
# repeat inside a repeat.
_MAX_HEADER_CHARS = 32 * 1024
_MAX_TRIGGER_WORDS = 6
_SECTION_SCAN_LINES = 8
_TRIGGER_KEYS = ("instance_prompt", "trigger_word")

_HEADER = re.compile(r"\A---[ \t]*\r?\n(?P<header>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]*>")
_EMPHASIS = re.compile(r"\*+|__|`")
_BOLD = re.compile(r"\*\*|__")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s*")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"\s+")
_QUOTES = "\"'“”‘’`"

_TRIGGER_LABEL = (
    r"\b(?:triggers?(?:\s+(?:words?|tokens?|phrases?|keywords?))?"
    r"|trained\s+words?|activation\s+(?:words?|tokens?|phrases?)|instance\s+prompt)\b"
)
_TRIGGER_HEADING = re.compile(_TRIGGER_LABEL, re.IGNORECASE)
_COLON = r"\s*[:=]\s*"
_QUOTED = r"(?:\"([^\"\n]+)\"|“([^”\n]+)”|'([^'\n]+)'|`([^`\n]+)`)"
# A quoted value needs no delimiter: ``Trigger word "SPROUT"``.
_TRIGGER_QUOTED = re.compile(rf"{_TRIGGER_LABEL}(?:{_COLON}|\s+){_QUOTED}", re.IGNORECASE)
_TRIGGER_AFTER_COLON = re.compile(rf"{_TRIGGER_LABEL}{_COLON}(?P<v>[^,;()\n\"`“'][^,;()\n]*)", re.IGNORECASE)
# "Trained words" lists the exact words the LoRA was trained with. A descriptive phrase there is the
# trigger, so it needs no code-like shape.
_TRAINED_WORDS = re.compile(r"\btrained\s+words?\s*[:=]\s*(?P<v>[^,\n]+)", re.IGNORECASE)
_NEGATED = re.compile(
    r"\b(?:no|without|never|doesn.?t\s+need|does\s+not\s+need)\b[^.\n]{0,25}\btrigger"
    r"|\btrigger[^.\n]{0,25}\b(?:not\s+(?:needed|required)|unnecessary)",
    re.IGNORECASE,
)
_NOT_A_TRIGGER = frozenset(
    {"none", "n/a", "na", "no", "not", "any", "use", "add", "include", "your", "the", "a", "an", "see", "for", "optional", "required", "none."}
)

_EXAMPLE_LABEL = (
    r"\b(?:(?:example|sample)\s+prompts?|prompts?\s+examples?|an?\s+example\s+of\s+(?:a\s+)?prompt)\b"
)
_EXAMPLE_HEADING = re.compile(_EXAMPLE_LABEL, re.IGNORECASE)
_EXAMPLE_LINE = re.compile(rf"{_EXAMPLE_LABEL}\s*[:=]?\s*(?P<v>[\"“'].+)$", re.IGNORECASE)

_DESCRIPTION_HEADING = re.compile(r"description|about|overview|introduction|summary|what", re.IGNORECASE)
_METADATA_LINE = re.compile(
    r"^(?:base\s+model|trained\s+words?|trigger(?:\s+\w+)?|activation\s+\w+|strength|license|"
    r"recommended\s+\w+|steps|seed|(?:example|sample)\s+prompts?|prompt\s+examples?)\s*[:=]",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"lora\s+finetune\s+of\s+model|generated\s+automatically|base\s+template\s+for\s+new\s+models",
    re.IGNORECASE,
)
_RULE = re.compile(r"^[-*_]{3,}$")


@dataclass(frozen=True, slots=True)
class CardSuggestions:
    trigger: str | None = None
    description: str = ""
    examples: tuple[str, ...] = ()


def _clean(text: str) -> str:
    return _SPACES.sub(" ", _CONTROL.sub("", text)).strip()


def _header_and_body(card: str) -> tuple[dict[str, object], str]:
    match = _HEADER.match(card)
    if match is None:
        return {}, _COMMENT.sub("", card)
    body = _COMMENT.sub("", card[match.end() :])
    header_text = match.group("header")[:_MAX_HEADER_CHARS]
    try:
        loaded: object = yaml.safe_load(header_text)
    except (yaml.YAMLError, ValueError, RecursionError):
        return {}, body
    if not isinstance(loaded, dict):
        return {}, body
    return cast("dict[str, object]", loaded), body


def _plain(line: str) -> str:
    """One line without bold marks, HTML tags and a list marker."""
    text = _HTML_TAG.sub("", _BOLD.sub("", line)).strip()
    return _LIST_MARKER.sub("", text).strip()


def _heading_text(line: str) -> str | None:
    """The text of a Markdown heading line, or ``None`` for any other line."""
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    return _plain(stripped.lstrip("#"))


def _trigger_value(raw: str) -> str | None:
    """A trigger candidate if it is short and not a sentence or a filler word."""
    value = _clean(raw).strip(_QUOTES + " ,.;:")
    if not value or len(value) > CUSTOM_LORA_TRIGGER_MAX:
        return None
    if len(value.split(" ")) > _MAX_TRIGGER_WORDS or value.lower() in _NOT_A_TRIGGER:
        return None
    if value.split(" ", 1)[0].lower() in _NOT_A_TRIGGER:
        return None
    return value


def _yaml_trigger(header: dict[str, object]) -> str | None:
    for key in _TRIGGER_KEYS:
        value = header.get(key)
        if isinstance(value, str):
            trigger = _trigger_value(value)
            if trigger is not None:
                return trigger
    return None


def _trigger_on_line(line: str) -> str | None:
    """A trigger named on one line: ``Trigger word: `X` ``, ``Trigger word "X"``, ``Trained words: X``."""
    if _NEGATED.search(line):
        return None
    trained = _TRAINED_WORDS.search(line)
    if trained is not None:
        value = _clean(trained.group("v")).strip(_QUOTES + " .;:")
        if value and len(value) <= CUSTOM_LORA_TRIGGER_MAX:
            return value
    quoted = _TRIGGER_QUOTED.search(line)
    if quoted is not None:
        trigger = _trigger_value(next(group for group in quoted.groups() if group is not None))
        if trigger is not None:
            return trigger
    after_colon = _TRIGGER_AFTER_COLON.search(line)
    if after_colon is not None:
        # A list of triggers: the first one is the main one.
        trigger = _trigger_value(after_colon.group("v").split(". ")[0])
        if trigger is not None:
            return trigger
    return None


def _section_trigger(line: str) -> str | None:
    """The value on one content line under a ``Trigger Word`` heading."""
    if line.startswith(("```", "|", ">")) or line.endswith(":"):
        return None
    span = _CODE_SPAN.search(line)
    if span is not None:
        return _trigger_value(span.group(1))
    return _trigger_value(line.split(",")[0])


def _body_trigger(body: str) -> str | None:
    in_section = False
    scanned = 0
    for raw in body.splitlines():
        heading = _heading_text(raw)
        if heading is not None:
            in_section = _TRIGGER_HEADING.search(heading) is not None and _NEGATED.search(heading) is None
            scanned = 0
            continue
        line = _plain(raw)
        if not line or line.startswith("|"):
            continue
        if in_section and scanned < _SECTION_SCAN_LINES:
            scanned += 1
            trigger = _section_trigger(line)
            if trigger is not None:
                return trigger
        trigger = _trigger_on_line(line)
        if trigger is not None:
            return trigger
    return None


def _widget_examples(header: dict[str, object]) -> list[str]:
    widget = header.get("widget")
    if not isinstance(widget, list):
        return []
    found: list[str] = []
    for entry in cast("list[object]", widget):
        text = cast("dict[str, object]", entry).get("text") if isinstance(entry, dict) else None
        if isinstance(text, str):
            _add_example(found, text)
    return found


def _add_example(found: list[str], text: str) -> None:
    example = _clean(text).strip(_QUOTES + " ")
    if example and len(example) <= CUSTOM_LORA_EXAMPLE_MAX and example not in found:
        found.append(example)


def _body_examples(body: str, found: list[str]) -> None:
    """Add ``Example prompt: "..."`` lines and the lines under an ``Example prompts`` heading."""
    in_section = False
    scanned = 0
    for raw in body.splitlines():
        if len(found) >= CUSTOM_LORA_EXAMPLES_MAX:
            return
        heading = _heading_text(raw)
        if heading is not None:
            in_section = _EXAMPLE_HEADING.search(heading) is not None
            scanned = 0
            continue
        line = _plain(raw)
        if not line or line.startswith(("```", "|", ">")):
            continue
        if in_section and scanned < _SECTION_SCAN_LINES:
            scanned += 1
            if not line.endswith(":"):
                _add_example(found, line)
            continue
        match = _EXAMPLE_LINE.search(line)
        if match is not None:
            value = match.group("v")
            # The example runs from the first quote to the last one on the line.
            _add_example(found, value[: max(value.rfind(quote) for quote in "\"”'") + 1] or value)


def _is_prose(line: str) -> bool:
    """Whether a line is a sentence. Images, tables, HTML, code fences and links alone are not."""
    return not line.startswith(("!", "|", "<", "```", "[![", "http")) and _RULE.match(line) is None


def _description(body: str) -> str:
    paragraph: list[str] = []
    for raw in body.splitlines():
        heading = _heading_text(raw)
        if heading is not None:
            if paragraph:
                break
            # A section heading before any prose ends the search, unless it introduces a description.
            if raw.strip().startswith("##") and _DESCRIPTION_HEADING.search(heading) is None:
                return ""
            continue
        line = raw.strip()
        if not line or not _is_prose(line):
            if paragraph:
                break
            continue
        text = _plain(line)
        if _METADATA_LINE.match(text):
            if paragraph:
                break
            continue
        if line.startswith(("- ", "* ")) and not paragraph:
            continue
        paragraph.append(_HTML_TAG.sub("", _MARKDOWN_LINK.sub(r"\1", text)))
    text = _clean(_EMPHASIS.sub("", " ".join(paragraph)))
    if _BOILERPLATE.search(text):
        return ""
    if len(text) <= CUSTOM_LORA_DESCRIPTION_MAX:
        return text
    cut = text[:CUSTOM_LORA_DESCRIPTION_MAX].rsplit(" ", 1)[0]
    return cut.rstrip(",;:")


def parse_model_card(card: str) -> CardSuggestions:
    """The trigger word, description and example prompts a card holds. Missing parts stay empty."""
    header, body = _header_and_body(card)
    examples = _widget_examples(header)
    _body_examples(body, examples)
    return CardSuggestions(
        trigger=_yaml_trigger(header) or _body_trigger(body),
        description=_description(body),
        examples=tuple(examples[:CUSTOM_LORA_EXAMPLES_MAX]),
    )
