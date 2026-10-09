/**
 * Downloads one asset through a URL that is valid now. A signed remote URL lives
 * 5 to 10 minutes, and a finished result stops polling, so the URL the page holds
 * can be dead. The asset is fetched again to get a new one. A failure goes to
 * `onError` and never throws, so a click handler can start it without a catch.
 */
export async function downloadAsset<TAsset extends { name: string }>({
  fetchAsset,
  mediaUrlForAsset,
  save,
  onError,
}: {
  fetchAsset: () => Promise<TAsset>;
  mediaUrlForAsset: (asset: TAsset) => string | null;
  save: (url: string, filename: string) => void;
  onError: () => void;
}): Promise<void> {
  try {
    const asset = await fetchAsset();
    const url = mediaUrlForAsset(asset);
    if (url === null) {
      onError();
      return;
    }
    save(url, asset.name);
  } catch {
    onError();
  }
}
