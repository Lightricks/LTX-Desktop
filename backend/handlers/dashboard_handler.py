from __future__ import annotations

from datetime import datetime

from _routes._errors import HTTPError
from services.dashboard_stats import (
    DashboardSnapshot,
    DashboardWindow,
    InvalidTimezone,
    LoraUsage,
)
from services.features.lora_recipes import LORA_RECIPES
from services.lora_catalog import LoraCatalogProvider
from services.records import UnavailableError
from services.store import Store


class DashboardHandler:
    def __init__(self, store: Store, catalog: LoraCatalogProvider) -> None:
        self._store = store
        self._catalog = catalog

    def get(
        self,
        *,
        window: DashboardWindow,
        tz: str,
        now: datetime | None = None,
    ) -> DashboardSnapshot:
        try:
            snapshot = self._store.dashboard(
                window=window,
                tz=tz,
                now=now,
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except InvalidTimezone as exc:
            raise HTTPError(422, "Invalid timezone", code="INVALID_TIMEZONE") from exc

        return snapshot.model_copy(
            update={
                "loras": snapshot.loras.model_copy(
                    update={"top": [self._named(usage) for usage in snapshot.loras.top]}
                )
            }
        )

    def _named(self, usage: LoraUsage) -> LoraUsage:
        """Runs made before names were stored carry the recipe id. The catalog still knows the name."""
        recipe = LORA_RECIPES.get(usage.feature)
        if recipe is None or usage.name != usage.feature:
            return usage
        item = self._catalog.get_lora(recipe.catalog_id)
        return usage if item is None else usage.model_copy(update={"name": item.name})
