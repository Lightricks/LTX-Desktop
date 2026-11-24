from __future__ import annotations

from typing import Protocol

from api_types import IcLoraCatalogItem, LoraCatalogFile, LoraCatalogItem


class LoraCatalogProvider(Protocol):
    def get_catalog(self) -> LoraCatalogFile: ...

    def get_ic_lora(self, ic_lora_id: str) -> IcLoraCatalogItem | None: ...

    def get_lora(self, lora_id: str) -> LoraCatalogItem | None: ...
