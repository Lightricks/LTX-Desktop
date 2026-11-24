export type AssetsLibraryBodyKind = "error" | "empty" | "loading" | "content";
export type AssetsInlineErrorKind = "next-page" | "refetch";
export type AssetsErrorRetryTarget = "refetch" | "fetchNextPage";

export type AssetsLibraryView = {
  kind: AssetsLibraryBodyKind;
  inlineError: AssetsInlineErrorKind | null;
  retryTarget: AssetsErrorRetryTarget;
};

export function resolveAssetsLibraryView({
  isLoading,
  error,
  loadedItemCount,
  isFetchNextPageError,
  hasNextPage = false,
}: {
  isLoading: boolean;
  error: unknown;
  loadedItemCount: number;
  isFetchNextPageError: boolean;
  hasNextPage?: boolean;
}): AssetsLibraryView {
  const isEmpty = loadedItemCount === 0;
  let kind: AssetsLibraryBodyKind;
  if (error && isEmpty) {
    kind = "error";
  } else if (isEmpty && (isLoading || hasNextPage)) {
    kind = "loading";
  } else if (isEmpty) {
    kind = "empty";
  } else {
    kind = "content";
  }

  if (kind !== "content" || !error) {
    return { kind, inlineError: null, retryTarget: "refetch" };
  }

  if (isFetchNextPageError) {
    return {
      kind: "content",
      inlineError: "next-page",
      retryTarget: "fetchNextPage",
    };
  }

  return {
    kind: "content",
    inlineError: "refetch",
    retryTarget: "refetch",
  };
}
