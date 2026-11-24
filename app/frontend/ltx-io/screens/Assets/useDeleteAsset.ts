import { useCallback, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { ApiResultError } from "@/ltx-io/lib/unwrapApiResult";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import { assetQueryKeys } from "../../hooks/assetQueryKeys";
import { generationQueryKeys } from "../../hooks/generationQueryKeys";
import {
  type AssetCacheSnapshot,
  optimisticallyRemoveAssetFromListCache,
  restoreAssetCache,
} from "./assetDeleteCache";

export function isAssetDeleteInUseError(error: unknown): boolean {
  return error instanceof ApiResultError && error.status === 409;
}

export function useDeleteAsset() {
  const { deleteAsset } = useExploreRuntime();
  const queryClient = useQueryClient();
  const deleteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const pendingAssetIdsRef = useRef<Set<string>>(new Set());
  const [pendingAssetIds, setPendingAssetIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  const syncPendingAssetIds = useCallback(() => {
    setPendingAssetIds(new Set(pendingAssetIdsRef.current));
  }, []);

  const mutation = useMutation<void, Error, string, AssetCacheSnapshot>({
    scope: { id: "asset-delete" },
    mutationFn: async (assetId) => {
      if (deleteAsset === null) {
        throw new Error("Asset delete is not available");
      }
      await deleteAsset(assetId);
    },
    onMutate: async (assetId) => {
      return optimisticallyRemoveAssetFromListCache(queryClient, assetId);
    },
    onError: (_error, _assetId, snapshots) => {
      if (snapshots) restoreAssetCache(queryClient, snapshots);
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: assetQueryKeys.all });
      void queryClient.invalidateQueries({ queryKey: generationQueryKeys.all });
    },
  });

  const mutateAsync = useCallback(
    (assetId: string): Promise<void> => {
      pendingAssetIdsRef.current.add(assetId);
      syncPendingAssetIds();

      const nextDelete = deleteQueueRef.current
        .then(() => mutation.mutateAsync(assetId))
        .finally(() => {
          pendingAssetIdsRef.current.delete(assetId);
          syncPendingAssetIds();
        });
      deleteQueueRef.current = nextDelete.catch(() => undefined);
      return nextDelete;
    },
    [mutation.mutateAsync, syncPendingAssetIds],
  );

  return { ...mutation, mutateAsync, pendingAssetIds };
}
