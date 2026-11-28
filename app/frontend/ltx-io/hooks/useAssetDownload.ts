import { createToast, DEFAULT_AUTOCLOSE_MS } from "../components/shared/Toast/toastService";
import { downloadAsset } from "../lib/downloadAsset";
import { downloadFromUrl } from "../lib/downloadFromUrl";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

import { fetchAsset } from "./generationQueryKeys";

export const DOWNLOAD_FAILED_MESSAGE = "The download failed. Try again.";

/** Starts the download of one asset in the remote app. A failure shows a toast. */
export function useAssetDownload(): (asset: { id: string }) => void {
  const { api, mediaUrlForAsset } = useExploreRuntime();

  return (asset) => {
    void downloadAsset({
      fetchAsset: () => fetchAsset(api, asset.id),
      mediaUrlForAsset,
      save: downloadFromUrl,
      onError: () =>
        createToast({
          message: DOWNLOAD_FAILED_MESSAGE,
          toastType: "danger",
          closeable: true,
          autoCloseAfter: DEFAULT_AUTOCLOSE_MS,
        }),
    });
  };
}
