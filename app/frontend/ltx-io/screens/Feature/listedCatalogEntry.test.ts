import assert from "node:assert/strict";
import { describe, it } from "node:test";

import catalog from "../../../../backend/runtime_config/lora_catalog.json" with { type: "json" };
import type { IcLoraListItem, LoraCatalogListItem } from "../../../hooks/use-catalog.ts";
import { listedIcLoraEntry, listedLoraEntry } from "./listedCatalogEntry.ts";

const icLora = catalog.ic_loras.find((entry) => entry.id === "day-to-night");
const lora = catalog.loras.find((entry) => entry.id === "cinemagraph-motion");
assert.ok(icLora && lora, "the shipped catalog must keep one two-variant IC-LoRA and one LoRA");

const [firstVariant, secondVariant] = icLora.download.variants;
assert.ok(firstVariant && secondVariant);

function icLoraItem(downloadedVariantIds?: string[]): IcLoraListItem {
  return {
    downloaded: downloadedVariantIds != null && downloadedVariantIds.length > 0,
    downloaded_variant_ids: downloadedVariantIds,
    ic_lora: icLora,
  } as unknown as IcLoraListItem;
}

function loraItem(downloadedVariantIds?: string[]): LoraCatalogListItem {
  return {
    downloaded: downloadedVariantIds != null && downloadedVariantIds.length > 0,
    downloaded_variant_ids: downloadedVariantIds,
    lora,
  } as unknown as LoraCatalogListItem;
}

const flags = (variants: { id: string; downloaded: boolean }[]) =>
  variants.map((variant) => [variant.id, variant.downloaded]);

describe("listedCatalogEntry", () => {
  it("reads the downloaded variant from a refreshed IC-LoRA entry, not the entry from before the download", () => {
    const before = listedIcLoraEntry(icLoraItem([]));
    const after = listedIcLoraEntry(icLoraItem([firstVariant.id]));
    assert.deepEqual(flags(before.variants), [
      [firstVariant.id, false],
      [secondVariant.id, false],
    ]);
    assert.deepEqual(flags(after.variants), [
      [firstVariant.id, true],
      [secondVariant.id, false],
    ]);
  });

  it("reads the downloaded variant from a refreshed LoRA entry, not the entry from before the download", () => {
    const [first, second] = lora.download.variants;
    assert.ok(first && second);
    const before = listedLoraEntry(loraItem([]));
    const after = listedLoraEntry(loraItem([second.id]));
    assert.deepEqual(flags(before.variants), [
      [first.id, false],
      [second.id, false],
    ]);
    assert.deepEqual(flags(after.variants), [
      [first.id, false],
      [second.id, true],
    ]);
  });

  it("marks no variant as downloaded when the API omits downloaded_variant_ids", () => {
    assert.deepEqual(
      listedIcLoraEntry(icLoraItem(undefined)).variants.filter((variant) => variant.downloaded),
      [],
    );
    assert.deepEqual(
      listedLoraEntry(loraItem(undefined)).variants.filter((variant) => variant.downloaded),
      [],
    );
  });
});
