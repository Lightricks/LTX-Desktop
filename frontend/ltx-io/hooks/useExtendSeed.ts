import { EXTEND_SEED_REVISION } from "./extendSeed.ts";
import { usePackagedSeedAsset } from "./usePackagedSeedAsset";

export function useExtendSeed(options?: { enabled?: boolean }) {
  return usePackagedSeedAsset("extend-video", {
    enabled: options?.enabled,
    revision: EXTEND_SEED_REVISION,
    media: "video",
  });
}
