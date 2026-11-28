import { useState } from "react";

import { useImageGenerationStatus } from "../../hooks/useImageGenerationStatus";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import {
  type ImageProvider,
  type ImageProviderOption,
  describeImageProviders,
  resolveImageProvider,
} from "./imageProviderAvailability";

/** Which provider a request uses, which can run now, and why a blocked one cannot. */
export function useImageProviderChoice(): {
  options: ImageProviderOption[];
  /** Null until the status loads. */
  provider: ImageProvider | null;
  /** The status request failed, so no provider can be chosen. */
  statusFailed: boolean;
  retryStatus: () => void;
  blockedReason: string | null;
  pick: (provider: ImageProvider) => void;
} {
  const runtime = useExploreRuntime();
  const statusQuery = useImageGenerationStatus();
  const status = statusQuery.data;
  const [picked, setPicked] = useState<ImageProvider | null>(null);
  const options = status ? describeImageProviders(status, runtime.host) : [];
  const provider = status ? resolveImageProvider(options, status.defaultProvider, picked) : null;
  return {
    options,
    provider,
    statusFailed: statusQuery.isError && status === undefined,
    retryStatus: () => void statusQuery.refetch(),
    blockedReason: options.find((option) => option.provider === provider)?.blockedReason ?? null,
    pick: setPicked,
  };
}
