"""Utilities for settings patch/load workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeGuard, cast

from services.services_utils import JSONValue

JSONObject: TypeAlias = dict[str, JSONValue]


def _is_json_value(value: object) -> TypeGuard[JSONValue]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        typed_items = cast(list[object], value)
        return all(_is_json_value(item) for item in typed_items)
    if isinstance(value, dict):
        typed_mapping = cast(dict[object, object], value)
        return all(isinstance(key, str) and _is_json_value(item) for key, item in typed_mapping.items())
    return False


def _is_json_object(value: object) -> TypeGuard[JSONObject]:
    if not isinstance(value, dict):
        return False
    typed_mapping = cast(dict[object, object], value)
    return all(isinstance(key, str) and _is_json_value(item) for key, item in typed_mapping.items())


def ensure_json_object(payload: object) -> JSONObject:
    if not _is_json_object(payload):
        raise ValueError("Settings payload must be a JSON object")
    return payload


def deep_merge_dicts(base: Mapping[str, JSONValue], patch: Mapping[str, JSONValue]) -> JSONObject:
    merged: JSONObject = dict(base)
    for key, value in patch.items():
        base_value = merged.get(key)
        if _is_json_object(value) and _is_json_object(base_value):
            merged[key] = deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def strip_none_values(payload: Mapping[str, JSONValue]) -> JSONObject:
    cleaned: JSONObject = {}
    for key, value in payload.items():
        if value is None:
            continue
        if _is_json_object(value):
            cleaned[key] = strip_none_values(value)
        else:
            cleaned[key] = value
    return cleaned


def collect_changed_paths(before: JSONValue, after: JSONValue, prefix: str = "") -> set[str]:
    if _is_json_object(before) and _is_json_object(after):
        paths: set[str] = set()
        for key in set(before.keys()) | set(after.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                paths.add(next_prefix)
                continue
            paths |= collect_changed_paths(before[key], after[key], next_prefix)
        return paths

    if before != after and prefix:
        return {prefix}
    return set()


def migrate_legacy_settings(raw: Mapping[str, JSONValue]) -> JSONObject:
    """Fold the per-conditioning API enhance gates back into one flag.

    The setting was split into T2V/I2V and is now a single control again. A saved file can
    hold either shape, and the split pair can disagree: OR keeps enhancement on for anyone
    who had it on for at least one conditioning, so nobody silently loses it on upgrade.
    """
    migrated: JSONObject = dict(raw)
    t2v = migrated.pop("prompt_enhancer_enabled_t2v", None)
    i2v = migrated.pop("prompt_enhancer_enabled_i2v", None)
    if (t2v is not None or i2v is not None) and "prompt_enhancer_enabled" not in migrated:
        migrated["prompt_enhancer_enabled"] = bool(t2v) or bool(i2v)

    return migrated
