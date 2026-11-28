import type { ExploreAsset } from "../../../../lib/explore-contract.ts";

/**
 * What a media field does with a freshly ingested asset: reject the wrong
 * kind, hand it to `prepareImport` (e.g. the trim gate), or accept it.
 * `accept` may receive a different (trimmed) asset, later or never.
 */
export async function applyImportedAsset({
  asset,
  isExpectedAsset,
  prepareImport,
  cacheAsset,
  accept,
  reject,
}: {
  asset: ExploreAsset;
  isExpectedAsset: (asset: ExploreAsset) => boolean;
  prepareImport?: (
    asset: ExploreAsset,
    accept: (asset: ExploreAsset) => void,
  ) => Promise<unknown>;
  cacheAsset: (asset: ExploreAsset) => void;
  accept: (asset: ExploreAsset) => void;
  reject: () => void;
}): Promise<void> {
  cacheAsset(asset);
  if (!isExpectedAsset(asset)) {
    reject();
    return;
  }
  if (prepareImport) {
    await prepareImport(asset, (accepted) => {
      cacheAsset(accepted);
      accept(accepted);
    });
    return;
  }
  accept(asset);
}
