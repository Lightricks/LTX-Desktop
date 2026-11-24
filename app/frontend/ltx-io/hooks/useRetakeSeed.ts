import { RETAKE_SEED_REVISION } from "./retakeSeed.ts";
import { usePackagedSeedAsset } from "./usePackagedSeedAsset";

export function useRetakeSeed(options?: { enabled?: boolean }) {
  return usePackagedSeedAsset("retake-video", {
    enabled: options?.enabled,
    revision: RETAKE_SEED_REVISION,
    media: "video",
  });
}
