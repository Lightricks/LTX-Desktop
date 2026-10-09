type LightboxItem = {
  id: string;
};

export type LightboxArrowKeyTarget = "content" | "chrome" | "blocked";

export function findLightboxAssetIndex(
  items: readonly LightboxItem[],
  activeAssetId: string,
): number {
  return items.findIndex((item) => item.id === activeAssetId);
}

export function nextLightboxIndex(
  index: number,
  loadedCount: number,
  hasNextPage: boolean,
  isFetchingNextPage: boolean,
): "next" | "fetch" | "end" {
  if (index + 1 < loadedCount) return "next";
  if (hasNextPage && !isFetchingNextPage) return "fetch";
  return "end";
}

export type LightboxDeepLinkAction = "idle" | "fetchNextPage" | "close";

export function resolveLightboxDeepLinkAction({
  lightboxAssetId,
  lightboxIndex,
  isLoading,
  isFetchingNextPage,
  isFetchNextPageError,
  hasNextPage,
}: {
  lightboxAssetId: string | null;
  lightboxIndex: number;
  isLoading: boolean;
  isFetchingNextPage: boolean;
  isFetchNextPageError: boolean;
  hasNextPage: boolean;
}): LightboxDeepLinkAction {
  if (lightboxAssetId === null || lightboxIndex >= 0) return "idle";
  if (isLoading || isFetchingNextPage) return "idle";
  if (isFetchNextPageError) return "idle";
  if (hasNextPage) return "fetchNextPage";
  return "close";
}

export function shouldHandleLightboxArrowKey(
  key: string,
  target: LightboxArrowKeyTarget,
  canNavigate: boolean,
): boolean {
  return (
    target !== "blocked" &&
    canNavigate &&
    (key === "ArrowLeft" || key === "ArrowRight")
  );
}

export function shouldOpenAssetFromKey(
  key: string,
  target: "tile" | "nested",
): boolean {
  return target === "tile" && (key === "Enter" || key === " ");
}

export function lightboxMediaKey(assetId: string, src: string | null): string {
  return `${assetId}\u0000${src ?? ""}`;
}
