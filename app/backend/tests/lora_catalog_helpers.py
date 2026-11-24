"""Catalog entries the prompt-enhancement tests inject into the fake catalog provider.

The shipped catalog is real data that changes underneath tests; these build throwaway entries
with only the field under test set, so a test reads as "an IC-LoRA with a trigger" rather than
as a full catalog record.
"""

from __future__ import annotations

from api_types import (
    DownloadSpec,
    DownloadVariant,
    IcLoraCatalogItem,
    InputSpec,
    LoraCatalogItem,
)


def download_spec(filename: str = "x.safetensors") -> DownloadSpec:
    return DownloadSpec(
        repo_id="org/x",
        variants=[DownloadVariant(id="default", label="Default", filename=filename, size_bytes=10, base_model="LTX-2.3")],
    )


def add_lora(fake_services, **overrides: object) -> LoraCatalogItem:
    defaults: dict[str, object] = dict(
        id="test-lora", name="Test Lora", description="d", download=download_spec(),
        requires_hf_login=False,
    )
    defaults.update(overrides)
    lora = LoraCatalogItem(**defaults)
    fake_services.lora_catalog_provider._catalog.loras.append(lora)
    return lora


def add_ic_lora(fake_services, **overrides: object) -> IcLoraCatalogItem:
    defaults: dict[str, object] = dict(
        id="test-ic-lora", name="Test IC-LoRA", description="d", download=download_spec(),
        requires_hf_login=False, input=InputSpec(kind="video"),
    )
    defaults.update(overrides)
    ic_lora = IcLoraCatalogItem(**defaults)
    fake_services.lora_catalog_provider._catalog.ic_loras.append(ic_lora)
    return ic_lora
