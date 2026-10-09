export type AssetsMediaKind = "image" | "video" | "audio";

export type AssetsListFilters = {
  mediaKind: AssetsMediaKind | null; // All
  sort: "created_at-desc" | "created_at-asc";
  q: string;
};

function isMediaKind(value: string): value is AssetsMediaKind {
  return value === "image" || value === "video" || value === "audio";
}

export function filtersFromSearchParams(params: URLSearchParams): AssetsListFilters {
  const mediaKindParam = params.get("media_kind");
  const sortParam = params.get("sort");
  return {
    mediaKind:
      mediaKindParam !== null && isMediaKind(mediaKindParam) ? mediaKindParam : null,
    sort: sortParam === "created_at-asc" ? "created_at-asc" : "created_at-desc",
    q: params.get("q") ?? "",
  };
}

export function searchParamsFromFilters(filters: AssetsListFilters): URLSearchParams {
  const params = new URLSearchParams();
  if (filters.mediaKind !== null) {
    params.set("media_kind", filters.mediaKind);
  }
  if (filters.sort === "created_at-asc") {
    params.set("sort", "created_at-asc");
  }
  if (filters.q !== "") {
    params.set("q", filters.q);
  }
  return params;
}

export function lightboxAssetIdFromSearchParams(params: URLSearchParams): string | null {
  const value = params.get("asset");
  return value !== null && value.length > 0 ? value : null;
}

export function searchParamsFromLibraryState(
  filters: AssetsListFilters,
  lightboxAssetId: string | null,
): URLSearchParams {
  const params = searchParamsFromFilters(filters);
  if (lightboxAssetId !== null) {
    params.set("asset", lightboxAssetId);
  }
  return params;
}
