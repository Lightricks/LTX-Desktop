"""Wildcard expansion for prompt templates.

Syntax: ``_wildcard_name_`` inside a prompt string is replaced by values
from a matching wildcard definition.  The parser supports:

* **All-combinations** expansion (Cartesian product of every wildcard slot).
* **Random-selection** expansion (pick *count* random fully-expanded prompts).
* **Nested wildcards** – a wildcard value may itself contain ``_other_``
  references which are recursively expanded.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# Pattern: underscore-delimited wildcard name, e.g. ``_color_``
_WILDCARD_RE = re.compile(r"_([A-Za-z][A-Za-z0-9_]*)_")


@dataclass(frozen=True)
class WildcardDef:
    """A single wildcard definition."""

    name: str
    values: list[str]


def _find_wildcard_names(prompt: str) -> list[str]:
    """Return unique wildcard names found in *prompt*, preserving first-seen order."""
    seen: set[str] = set()
    names: list[str] = []
    for m in _WILDCARD_RE.finditer(prompt):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _replace_single(prompt: str, name: str, value: str) -> str:
    """Replace all occurrences of ``_name_`` in *prompt* with *value*."""
    return prompt.replace(f"_{name}_", value)


def _resolve_nested(
    prompt: str,
    lookup: dict[str, list[str]],
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Recursively expand wildcards in *prompt*.

    Returns a list of all combinations produced by every wildcard slot.
    """
    if depth > max_depth:
        return [prompt]

    names = _find_wildcard_names(prompt)
    if not names:
        return [prompt]

    # Expand the first wildcard found, then recurse on each variant.
    first = names[0]
    values = lookup.get(first, [f"_{first}_"])  # keep literal if undefined
    results: list[str] = []
    for val in values:
        replaced = _replace_single(prompt, first, val)
        results.extend(_resolve_nested(replaced, lookup, depth + 1, max_depth))
    return results


def _build_lookup(wildcards: list[WildcardDef]) -> dict[str, list[str]]:
    return {w.name: w.values for w in wildcards}


def expand_prompt(prompt: str, wildcards: list[WildcardDef]) -> list[str]:
    """Return **all** expanded combinations of *prompt* with the given wildcards.

    Example::

        expand_prompt(
            "A _color_ _animal_",
            [WildcardDef("color", ["red", "blue"]),
             WildcardDef("animal", ["cat", "dog"])],
        )
        # → ["A red cat", "A red dog", "A blue cat", "A blue dog"]
    """
    lookup = _build_lookup(wildcards)
    return _resolve_nested(prompt, lookup)


def expand_random(
    prompt: str,
    wildcards: list[WildcardDef],
    count: int = 1,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Return *count* randomly-expanded variants of *prompt*.

    Each returned string has every wildcard slot filled by a random value
    drawn independently.  Duplicates are possible when *count* exceeds the
    number of unique combinations.
    """
    _rng = rng or random.Random()
    lookup = _build_lookup(wildcards)

    results: list[str] = []
    for _ in range(count):
        text = prompt
        for _ in range(20):  # bounded recursion for nested wildcards
            names = _find_wildcard_names(text)
            if not names:
                break
            for name in names:
                values = lookup.get(name)
                if values:
                    text = _replace_single(text, name, _rng.choice(values))
        results.append(text)
    return results
