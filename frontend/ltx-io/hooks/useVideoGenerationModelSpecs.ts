import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import {
  fetchVideoGenerationModelSpecs,
  generationQueryKeys,
  VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS,
} from "./generationQueryKeys";
import { useInvalidateOnGenerationFinish } from "./useInvalidateOnGenerationFinish";

export function useVideoGenerationModelSpecs() {
  const { api, modelsVersion, budgetMode, budgetModeSaving, neverSplitIntoParts, neverSplitSaving } =
    useExploreRuntime();
  // A clean local run can widen what this device has proven, unlocking a Cautious cell.
  useInvalidateOnGenerationFinish(generationQueryKeys.allSpecs);
  const query = useQuery({
    queryKey: generationQueryKeys.specs(
      modelsVersion,
      budgetMode ?? "server",
      neverSplitIntoParts ?? "server",
    ),
    queryFn: () => fetchVideoGenerationModelSpecs(api),
    placeholderData: keepPreviousData,
    ...VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS,
  });
  // Generate waits until the specs are priced with the saved budget settings.
  const budgetSettingsReady =
    (budgetMode == null || (!budgetModeSaving && query.data?.budgetMode === budgetMode)) &&
    (neverSplitIntoParts == null ||
      (!neverSplitSaving && query.data?.neverSplitIntoParts === neverSplitIntoParts));
  return {
    data: query.data,
    error: query.error,
    isError: query.isError,
    isFetching: query.isFetching,
    isPending: query.isPending,
    isSuccess: query.isSuccess,
    refetch: query.refetch,
    status: query.status,
    budgetSettingsReady,
  };
}
