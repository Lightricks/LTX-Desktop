export function canonicalAssetSearchQuery(q: string | undefined): string | undefined {
  const normalized = q?.trim();
  return normalized === "" ? undefined : normalized;
}

export const assetQueryKeys = {
  all: ["assets"] as const,
  list: (filters: { media_kind?: string; sort: string; q?: string }) => {
    const q = canonicalAssetSearchQuery(filters.q);
    const normalizedFilters = { ...filters };
    if (q === undefined) delete normalizedFilters.q;
    else normalizedFilters.q = q;
    return [...assetQueryKeys.all, "list", normalizedFilters] as const;
  },
  detail: (assetId: string) => [...assetQueryKeys.all, "detail", assetId] as const,
};

export function isImageAsset(asset: { media_kind: string }): boolean {
  return asset.media_kind === "image";
}

export function isAudioAsset(asset: { media_kind: string }): boolean {
  return asset.media_kind === "audio";
}

export function isVideoAsset(asset: { media_kind: string }): boolean {
  return asset.media_kind === "video";
}
