import { useEffect, useState } from "react";

import type { ExploreListedAsset } from "@/lib/explore-contract";

import { stillIdAtPosition, stillPosition } from "./keyframesModel";

/**
 * The big preview of one still, which steps through all the stills. It keeps the still, not its
 * place: the place in `items` moves when an earlier asset finishes loading, so the place is found
 * again at each render. `lightboxProps` is null while no preview is open.
 */
export function useKeyframePreview(
  orderedIds: readonly string[],
  lightbox: { items: ExploreListedAsset[]; positions: (number | null)[] },
): {
  open: (id: string) => void;
  lightboxProps: {
    items: ExploreListedAsset[];
    activeAssetId: string;
    activeIndex: number;
    onActiveIndexChange: (position: number) => void;
    onClose: () => void;
  } | null;
} {
  const [previewId, setPreviewId] = useState<string | null>(null);
  const position =
    previewId === null ? null : stillPosition(orderedIds, lightbox.positions, previewId);
  const item = position === null ? undefined : lightbox.items[position];
  // The previewed still was removed, or its asset is gone. Do not show another still later.
  useEffect(() => {
    if (previewId !== null && item === undefined) setPreviewId(null);
  }, [previewId, item]);

  return {
    open: setPreviewId,
    lightboxProps:
      position === null || !item
        ? null
        : {
            items: lightbox.items,
            activeAssetId: item.id,
            // One image can sit at two times, so the place picks the still, not the id.
            activeIndex: position,
            onActiveIndexChange: (next) => {
              const id = stillIdAtPosition(orderedIds, lightbox.positions, next);
              if (id !== undefined) setPreviewId(id);
            },
            onClose: () => setPreviewId(null),
          },
  };
}
