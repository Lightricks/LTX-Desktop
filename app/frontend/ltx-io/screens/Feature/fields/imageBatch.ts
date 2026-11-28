import type { ExploreAsset } from "@/lib/explore-contract";

import { imageIngestErrorMessage, UNSUPPORTED_IMAGE_MESSAGE } from "./imageAssetInput.ts";

/** Keep the first `slotsLeft` items, in order. `overLimit` counts the items cut. */
export function planImageBatch<T>(
  items: readonly T[],
  slotsLeft: number,
): { take: T[]; overLimit: number } {
  const room = Math.max(0, slotsLeft);
  return { take: items.slice(0, room), overLimit: Math.max(0, items.length - room) };
}

/**
 * Split the settled ingests into the images to add and the problems to show.
 * One failed or wrong-kind file never removes the others. Order is the pick order.
 */
export function collectImageBatch(
  results: readonly PromiseSettledResult<ExploreAsset>[],
  isExpectedAsset: (asset: ExploreAsset) => boolean,
): { assets: ExploreAsset[]; wrongKind: number; failures: unknown[] } {
  const assets: ExploreAsset[] = [];
  const failures: unknown[] = [];
  let wrongKind = 0;
  for (const result of results) {
    if (result.status === "rejected") {
      failures.push(result.reason);
    } else if (isExpectedAsset(result.value)) {
      assets.push(result.value);
    } else {
      wrongKind += 1;
    }
  }
  return { assets, wrongKind, failures };
}

/** One message for the whole batch, or null when every file was added. */
export function describeImageBatch({
  unsupported,
  overLimit,
  maxCount,
  failures,
}: {
  unsupported: number;
  overLimit: number;
  maxCount: number;
  failures: readonly unknown[];
}): string | null {
  const parts: string[] = [];
  if (failures.length > 0) parts.push(imageIngestErrorMessage(failures[0]));
  if (unsupported > 0) parts.push(UNSUPPORTED_IMAGE_MESSAGE);
  if (overLimit > 0) parts.push(`The limit is ${maxCount} images. ${overLimit} not added.`);
  return parts.length > 0 ? parts.join(" ") : null;
}
