from __future__ import annotations

import logging
from pathlib import Path

import requests

from api_types import IcLoraCatalogItem, LoraCatalogFile, LoraCatalogItem, parse_lora_catalog

logger = logging.getLogger(__name__)


class FileLoraCatalogProvider:
    """Reads the LoRA / IC-LoRA catalog from a local path or an http(s) URL.

    When ``source`` is a URL and the fetch fails, falls back to a bundled file so the
    library keeps working offline / when the remote is down.
    """

    def __init__(self, source: str, fallback_path: str | None = None) -> None:
        self._source = source
        self._fallback_path = fallback_path
        self._cache: LoraCatalogFile | None = None

    def get_catalog(self) -> LoraCatalogFile:
        if self._cache is None:
            self._cache = self._load()
        return self._cache

    def get_ic_lora(self, ic_lora_id: str) -> IcLoraCatalogItem | None:
        return next((r for r in self.get_catalog().ic_loras if r.id == ic_lora_id), None)

    def get_lora(self, lora_id: str) -> LoraCatalogItem | None:
        return next((r for r in self.get_catalog().loras if r.id == lora_id), None)

    def _load(self) -> LoraCatalogFile:
        if not self._source.startswith(("http://", "https://")):
            return parse_lora_catalog(Path(self._source).read_text(encoding="utf-8"))

        # Split fetch failure from parse failure: a network blip falls back to the bundled
        # copy (offline resilience), but a *fetched* remote that fails to parse is logged
        # loudly and still falls back — so a bad remote can't take the library offline, yet
        # it's never silent (a green fetch + red parse means the remote, not the network).
        try:
            resp = requests.get(self._source, timeout=15)
            resp.raise_for_status()
            text = resp.text
        except requests.RequestException:
            if self._fallback_path is None:
                raise
            logger.warning("Remote LoRA catalog fetch failed; using bundled fallback")
            return parse_lora_catalog(Path(self._fallback_path).read_text(encoding="utf-8"))

        try:
            return parse_lora_catalog(text)
        except Exception:
            if self._fallback_path is None:
                raise
            logger.error(
                "Remote LoRA catalog fetched but failed to parse; using bundled fallback "
                "(remote update not applied — check the remote schema)",
                exc_info=True,
            )
            return parse_lora_catalog(Path(self._fallback_path).read_text(encoding="utf-8"))
