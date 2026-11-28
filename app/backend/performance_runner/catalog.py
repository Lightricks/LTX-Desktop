"""SSOT client for the app's LoRA / IC-LoRA catalog.

Resolves a scenario's catalog *id* against the app's library — /api/loras +
/api/ic-loras, backed by the same runtime_config/lora_catalog.json the app ships —
which report each item's download filename and on-disk `downloaded` status. A scenario
names an id; its weight ref (``loras/<id>/<file>``) and availability come from here, the
one place the app already owns them.

Fetched once per process and cached; call refresh() (e.g. from the dashboard's
refresh button) to pick up a LoRA downloaded mid-session. If the backend can't be
reached, availability is reported as "unknown" (empty missing-list) so scenarios
still run and surface the real backend error, mirroring models_dir()'s behaviour.
"""

from __future__ import annotations

import perf_config

# id -> {"downloaded": bool, "filename": str | None, "name": str}
_LORAS: dict[str, dict[str, object]] | None = None
_IC_LORAS: dict[str, dict[str, object]] | None = None
_reachable = True


def _index(items: object, item_key: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        obj = it.get(item_key) or {}
        cid = obj.get("id") if isinstance(obj, dict) else None
        if not isinstance(cid, str):
            continue
        dl = obj.get("download") or {}
        filename = None
        if isinstance(dl, dict):
            variants = dl.get("variants")
            if isinstance(variants, list) and variants and isinstance(variants[0], dict):
                # First variant is the catalog default checkpoint.
                fn = variants[0].get("filename")
                filename = fn if isinstance(fn, str) else None
        out[cid] = {
            "downloaded": bool(it.get("downloaded")),
            "filename": filename,
            "name": obj.get("name") or cid,
        }
    return out


def _fetch(path: str, list_key: str, item_key: str) -> dict[str, dict[str, object]]:
    global _reachable
    try:
        data = perf_config._get(path) or {}  # noqa: SLF001
    except Exception:  # noqa: BLE001
        _reachable = False
        return {}
    return _index(data.get(list_key) if isinstance(data, dict) else None, item_key)


def _loras() -> dict[str, dict[str, object]]:
    global _LORAS
    if _LORAS is None:
        _LORAS = _fetch("/api/loras", "loras", "lora")
    return _LORAS


def _ic_loras() -> dict[str, dict[str, object]]:
    global _IC_LORAS
    if _IC_LORAS is None:
        _IC_LORAS = _fetch("/api/ic-loras", "ic_loras", "ic_lora")
    return _IC_LORAS


def refresh() -> None:
    """Drop the cache so the next lookup re-fetches (e.g. after a mid-session download)."""
    global _LORAS, _IC_LORAS, _reachable
    _LORAS = _IC_LORAS = None
    _reachable = True


def lora_ref(lora_id: str) -> str | None:
    """Generation-payload `ref` for a catalog LoRA: ``loras/<id>/<filename>`` (mirrors
    runtime_config.resolve_lora_path). None if the id isn't in the catalog."""
    info = _loras().get(lora_id)
    fn = info.get("filename") if info else None
    return f"loras/{lora_id}/{fn}" if fn else None


def unavailable(required_loras: list[str], required_ic_loras: list[str]) -> list[str]:
    """Display names of required catalog items that are NOT downloaded (empty = all
    present). An unknown id (not in the catalog) is reported by its id. When the backend
    is unreachable, returns [] so the scenario still runs and surfaces the real error."""
    loras, ic_loras = _loras(), _ic_loras()
    if not _reachable:
        return []
    missing: list[str] = []
    for cid in required_loras:
        info = loras.get(cid)
        if not info or not info["downloaded"]:
            missing.append(str(info["name"]) if info else cid)
    for cid in required_ic_loras:
        info = ic_loras.get(cid)
        if not info or not info["downloaded"]:
            missing.append(str(info["name"]) if info else cid)
    return missing
